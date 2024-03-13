import mindspore
import mindspore.context as context
import x2ms_adapter
from x2ms_adapter.core.context import x2ms_context
from x2ms_adapter.core.cell_wrapper import WithLossCell
from x2ms_adapter.torch_api.optimizers import optim_register
from x2ms_adapter.core.exception import TrainBreakException, TrainContinueException, TrainReturnException
from x2ms_adapter.torch_api.optimizers import optim_register
import mindspore
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.loss as loss_wrapper
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
import numpy as np
if not x2ms_context.is_context_init:
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
    x2ms_context.is_context_init = True
import argparse
from data_process import Dataset
from model10 import HyperNet
import numpy as np
import math
from tester import Tester
import os
import json


def save_model(model, opt, measure, args, measure_by_arity=None, test_by_arity=False, itr=0, test_or_valid='test', is_best_model=False):
    """
    Save the model state to the output folder.
    If is_best_model is True, then save the model also as best_model.chkpnt
    """
    if is_best_model:
        x2ms_adapter.save(x2ms_adapter.nn_cell.state_dict(model), os.path.join(args.output_dir, 'best_model.chkpnt'))
        print("######## Saving the BEST MODEL")

    model_name = 'model_{}itr.chkpnt'.format(itr)
    opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(args.model)
    measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(args.model)
    print("######## Saving the model {}".format(os.path.join(args.output_dir, model_name)))

    x2ms_adapter.save(x2ms_adapter.nn_cell.state_dict(model), os.path.join(args.output_dir, model_name))
    x2ms_adapter.save(x2ms_adapter.nn_cell.state_dict(opt), os.path.join(args.output_dir, opt_name))
    if measure is not None:
        measure_dict = vars(measure)
        # If a best model exists
        if is_best_model:
            measure_dict["best_iteration"] = x2ms_adapter.tensor_api.item(model.best_itr)
            measure_dict["best_mrr"] = x2ms_adapter.tensor_api.item(model.best_mrr)
        with open(os.path.join(args.output_dir, measure_name), 'w') as f:
            json.dump(measure_dict, f, indent=4, sort_keys=True)
    # Note that measure_by_arity is only computed at test time (not validation)
    if (test_by_arity) and (measure_by_arity is not None):
        H = {}
        measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid,
                                                                        itr) if itr else '{}.json'.format(
            args.model)
        for key in measure_by_arity:
            H[key] = vars(measure_by_arity[key])
        with open(os.path.join(args.output_dir, measure_by_arity_name), 'w') as f:
            json.dump(H, f, indent=4, sort_keys=True)


def decompose_predictions(targets, predictions, max_length):
    positive_indices = np.where( targets > 0)[0]
    seq = []
    for ind, val in enumerate(positive_indices):
        if (ind == len(positive_indices) - 1):
            seq.append(padd(predictions[val:], max_length))
        else:
            seq.append(padd(predictions[val:positive_indices[ind + 1]], max_length))
    return seq


def padd(a, max_length):
    b = mindspore.ops.pad(a, (0, max_length - len(a)), 'constant')
    return b


def padd_and_decompose(targets, predictions, max_length):
    seq = decompose_predictions(targets, predictions, max_length)
    return mindspore.ops.stack(seq)

def main(args):
    # args.arity_lst = [2, 4, 5]
    args.arity_lst = [2, 4, 5]
    # args.arity_lst = [4]
    # args.arity_lst = [4]
    # args.arity_lst = [2, 3, 4, 5]
    # args.arity_lst = [2, 3, 4, 5, 6]
    max_arity = args.arity_lst[-1]
    args.device = x2ms_adapter.Device("cuda:0" if x2ms_adapter.is_cuda_available() else "cpu")
    dataset = Dataset(data_dir=args.dataset, arity_lst=args.arity_lst, device=args.device)
    model = HyperNet(dataset, emb_dim=args.emb_dim, hidden_drop=args.hidden_drop)
    # opt = optim_register.adagrad(x2ms_adapter.parameters(model), lr=args.lr, weight_decay=5e-6)
    opt = mindspore.nn.Adagrad(model.trainable_params(), learning_rate=args.lr, weight_decay=5e-6,accum=0.0,update_slots=False)

    for name, param in x2ms_adapter.named_parameters(model):
        print('Parameter %s: %s, require_grad = %s' % (name, str(x2ms_adapter.tensor_api.x2ms_size(param)), str(param.requires_grad)))
    # If the number of iterations is the same as the current iteration, exit.
    if (model.cur_itr.data >= args.num_iterations):
        print("*************")
        print("Number of iterations is the same as that in the pretrained model.")
        print("Nothing left to train. Exiting.")
        print("*************")
        return

    print("Training the {} model...".format(args.model))
    print("Number of training data points: {}".format(dataset.num_ent))

    # loss_layer = loss_wrapper.CrossEntropyLoss()
    loss_layer = mindspore.nn.CrossEntropyLoss()
    print("Starting training at iteration ... {}".format(model.cur_itr.data))
    test_by_arity = args.test_by_arity
    best_model = None
    for it in range(model.cur_itr.data, args.num_iterations + 1):
        x2ms_adapter.x2ms_train(model)
        model.cur_itr.data += 1
        losses = 0
        def construct(arity):
            nonlocal losses
            losses = losses if 'losses' in locals().keys() else None
            arity = arity
            last_batch = False
            while not last_batch:
                batch, ms, bs = dataset.next_batch(args.batch_size, args.nr, arity, args.device)
                targets = x2ms_adapter.tensor_api.numpy(batch[:, -2])
                batch = batch[:, :-2]
                last_batch = dataset.is_last_batch()
                x2ms_adapter.nn_cell.zero_grad(opt)
                number_of_positive = len(x2ms_adapter.tensor_api.where(np, targets > 0)[0])

                def forward_fn(batch, ms, bs, targets):
                    predictions = model.construct(batch, ms, bs)
                    # predictions = model.forward(batch)
                    predictions = padd_and_decompose(targets, predictions, args.nr * max_arity)
                    targets = x2ms_adapter.zeros(number_of_positive).int()
                    loss = loss_layer(predictions, targets)
                    return loss, predictions

                grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)
                (loss, _), grads = grad_fn(batch, ms, bs, targets)
                opt(grads)
                # predictions = model.construct(batch, ms, bs)
                # # predictions = model.forward(batch)
                # predictions = padd_and_decompose(targets, predictions, args.nr * max_arity)
                # targets = x2ms_adapter.zeros(number_of_positive).int()
                # loss = loss_layer(predictions, targets)
                # print(loss)
                if math.isnan(loss):
                    print(loss)
                    continue
                # opt.step()
                losses += x2ms_adapter.tensor_api.item(loss)
            # self.set_output(batch, bs, last_batch, loss, ms, number_of_positive, predictions, targets)

        # wrapped_model = WithLossCell(construct=construct, key='times_0')
        # wrapped_model = x2ms_adapter.train_one_step_cell(wrapped_model, optim_register.get_instance())
        for arity in args.arity_lst:
            construct(arity)
            # wrapped_model.network._input = (arity)
            # wrapped_model()
            # batch, bs, last_batch, loss, ms, number_of_positive, predictions, targets = wrapped_model.network.output
        print("Iteration#: {}, loss: {}".format(it, losses))
        if (it % 50 == 0 and it != 0) or (it == args.num_iterations):
            print("validation:")
            tester = Tester(dataset, model, "valid", args.model)
            measure_valid, _ = tester.test()
            mrr = measure_valid.mrr["fil"]
            is_best_model = (best_model is None) or (mrr > best_model.best_mrr)
            if is_best_model:
                best_model = model
                # Update the best_mrr value
                best_model.best_mrr=mindspore.Parameter(mindspore.Tensor(np.array([mrr]), dtype=mindspore.float64), requires_grad=False)
                best_model.best_itr=mindspore.Parameter(mindspore.Tensor(np.array([it]), dtype=mindspore.int32), requires_grad=False)
    tester = Tester(dataset, best_model, "test", args.model)
    # measure_all, _ = tester.test(test_by_arity=False)
    # save_model(best_model, opt, measure_all, args, test_by_arity=False, itr=best_model.cur_itr, test_or_valid="test")
    measure_arity, measure_by_arity = tester.test(test_by_arity=test_by_arity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HyperNet")
    parser.add_argument('-dataset', type=str, default="../../../data/FB-AUTO")
    parser.add_argument('-lr', type=float, default=0.05)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-filt_w', type=int, default=1)
    parser.add_argument('-filt_h', type=int, default=1)
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-input_drop', type=float, default=0.2)
    parser.add_argument('-stride', type=int, default=2)
    parser.add_argument('-num_iterations', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-test_by_arity', type=bool, default=True)
    parser.add_argument("-test", action="store_true",
                        help="If -test is set, then you must specify a -pretrained model. "
                             + "This will perform testing on the pretrained model and save the output in -output_dir")
    parser.add_argument('-pretrained', type=str, default=None,
                        help="A path to a trained model (.chkpnt file), which will be loaded if provided.")
    parser.add_argument('-output_dir', type=str, default="./record/",
                        help="A path to the directory where the model will be saved and/or loaded from.")
    parser.add_argument('-restartable', action="store_true",
                        help="If restartable is set, you must specify an output_dir")
    args = parser.parse_args()

    main(args)
