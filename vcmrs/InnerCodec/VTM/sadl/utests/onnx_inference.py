import onnxruntime
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    prog="onnx_inference and dump", usage="NB: force run on CPU"
)
parser.add_argument(
    "--input_onnx", action="store", nargs="?", type=str, help="name of the onnx file"
)
parser.add_argument(
    "--output", action="store", nargs="?", type=str, help="name of results file"
)
parser.add_argument(
    "--no_transpose",
    action="store_true",
    default=False,
    help="do not transpose input/output",
)
parser.add_argument(
    "--input_default_value",
    action="store",
    nargs="?",
    type=int,
    default=8,
    help="default values to replace named inputs",
)
args = parser.parse_args()
if args.input_onnx is None:
    quit("[ERROR] You should specify an onnx file")


session = onnxruntime.InferenceSession(args.input_onnx, None)
inputs = {}
v = []
for n in session.get_inputs():
    L = n.shape
    for i in range(len(L)):
        if type(L[i]) is not int:
            L[i] = args.input_default_value
    inputs[n.name] = np.random.uniform(size=L).astype("float32")
    v.append(inputs[n.name])
outputs = []
for n in session.get_outputs():
    outputs.append(n.name)
result = session.run(outputs, inputs)

if args.output is not None:
    with open(args.output, "w") as f:
        f.write(str(len(inputs)) + "\n")
        for input in v:
            # print(input.shape)
            if input.shape[0] != 1:
                raise ("[ERROR] inputs should include a batch size of size=1.")
            if len(input.shape) <= 1 or input.shape[0] != 1:
                print("[WARN] inputs should include batch size of size=1.")
                f.write("2" + "\n" + "1 ")  # force batch 1
            else:
                f.write(str(len(input.shape)) + "\n")
                if not args.no_transpose:
                    if len(input.shape) == 4:
                        input = np.transpose(input, (0, 2, 3, 1)).copy()  # nchw to nhwc
            for i in input.shape:
                f.write("{} ".format(i))
            f.write("\n")
            for x in np.nditer(input):
                f.write("{} ".format(x))
            f.write("\n")
        f.write("{}\n".format(len(outputs)))
        for o in result:
            f.write(str(len(o.shape)) + "\n")
            if not args.no_transpose:
                if len(o.shape) == 4:
                    o = np.transpose(o, (0, 2, 3, 1)).copy()  # nchw to nhwc
                if len(o.shape) == 3:
                    o = np.transpose(o, (1, 2, 0)).copy()  # chw to hwc
            for i in o.shape:
                f.write("{} ".format(i))
            f.write("\n")
            for x in np.nditer(o):
                if np.issubdtype(x.dtype, np.bool_):
                    x = x.astype(float)
                f.write("{} ".format(x))
            f.write("\n")
        print("[INFO] results file in {}".format(args.output))
