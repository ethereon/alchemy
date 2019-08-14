"""
Utilities for porting / validating parameters from Google Research's
reference implementation of DeepLab v3.

This is primarily intended for development / debugging.
"""


import fire
import numpy as np
import tensorflow as tf

from merlin.context import active_context
from merlin.snapshot import Snapshot
from merlin.shape import Axis

from alchemy.networks.xception import Xception, Xception65
from alchemy.models.deeplab import Predictor, DeepLab

# TensorFlow v1 compatibility interface for legacy frozen reference models
tfc = tf.compat.v1


class XceptionDefaults:
    INPUT_NAME = 'ImageTensor:0'
    OUTPUT_NAME = 'xception_65/exit_flow/block2/unit_1/xception_module/separable_conv3_pointwise/Relu:0'

    @staticmethod
    def create():
        return Xception(**Xception65(output_stride=8))


class DeepLabDefaults:
    INPUT_NAME = 'ImageTensor:0'
    OUTPUT_NAME = 'logits/semantic/BiasAdd:0'

    @staticmethod
    def create():
        PASCAL_CLASS_COUNT = 21
        return Predictor(config=DeepLab(logit_channels=PASCAL_CLASS_COUNT))


Defaults = DeepLabDefaults


def get_test_input():
    """
    Return a deterministic random input tensor.
    """
    np.random.seed(0xCafeBeef)
    return (np.random.rand(1, 513, 513, 3) * 255).astype(np.uint8).astype(np.float32)


def preprocess(tensor):
    """
    Pre-process input to match DeepLab v3
    """
    return (tensor * 2. / 255.) - 1.0


def load_model(model_path):
    """
    Load a legacy TensorFlow frozen graph.
    """
    graph = tfc.Graph()
    with graph.as_default():
        od_graph_def = tfc.GraphDef()
        with tfc.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tfc.import_graph_def(od_graph_def, name='')
    return graph


def initialize_model(ckpt_path, mapping_csv_path, model_factory=Defaults.create):
    """
    Initialize the new model from a reference checkpoint.
    """
    # Load the reference checkpoint
    ckpt = tfc.train.NewCheckpointReader(ckpt_path)

    # Get the source -> destination mapping
    with open(mapping_csv_path, 'r') as csv_file:
        mapping = [line.strip().split(',') for line in csv_file.readlines()]

    # Create the model + perform a forward pass to create the vars
    model = model_factory()
    model(get_test_input())

    # Create a LUT for the destination variables
    dst_lut = {var.name: var for var in model.variables}

    # Transfer weights
    for (src, dst) in mapping:
        dst_lut[dst].assign(ckpt.get_tensor(src))

    return model


def generate_reference_output(model,
                              output,
                              input_tensor_name=Defaults.INPUT_NAME,
                              output_tensor_name=Defaults.OUTPUT_NAME):
    """
    Save the output of the reference model.
    """
    graph = load_model(model)
    with tfc.Session(graph=graph) as sess:
        out = sess.run(
            graph.get_tensor_by_name(output_tensor_name),
            feed_dict={input_tensor_name: get_test_input()}
        )
        np.save(output, out)


def diff_against_reference(ckpt, mapping, reference):
    """
    Compare the output of the new model with a reference.
    """
    print('Transferring weights')
    model = initialize_model(ckpt_path=ckpt, mapping_csv_path=mapping)

    print('Evaluating')
    our_output = model(preprocess(get_test_input()))

    ref_output = np.load(reference)
    print(f'Max absolute delta: {np.max(np.abs(ref_output - our_output))}')

    ref_argmax = np.argmax(ref_output, axis=Axis.channel)
    our_argmax = np.argmax(our_output, axis=Axis.channel)
    argmax_mismatch = np.sum(ref_argmax != our_argmax)
    print(f'Argmax mismatch: {argmax_mismatch}')


def list_variable_names(sort=False, model_factory=Defaults.create):
    """
    Lists all variable names.
    """
    # Create the model and invoke it to create all variables
    model = model_factory()
    model(get_test_input())
    # Get all variable names and maybe sort 'em
    names = [var.name for var in model.variables]
    if sort:
        names.sort()
    # Display variable names
    print('\n'.join(names))


def export_snapshot(ckpt, mapping, snapshot_dir):
    print('Transferring weights')
    model = initialize_model(ckpt_path=ckpt, mapping_csv_path=mapping)
    print('Exporting snapshot')
    snapshot = Snapshot(
        config=Snapshot.Config(directory=snapshot_dir),
        model=model
    )
    snapshot.save()


def main():
    print(f'Using {Defaults.__name__}')
    with active_context.in_inference_mode():
        fire.Fire({
            'diff': diff_against_reference,
            'gen-ref': generate_reference_output,
            'list': list_variable_names,
            'export': export_snapshot
        })


if __name__ == "__main__":
    main()
