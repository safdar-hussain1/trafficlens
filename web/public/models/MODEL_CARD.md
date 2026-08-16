# yolo11n-480.onnx

The detector the demo page runs on your own device. Nothing about a frame you
load leaves your browser: this file is downloaded once, cached, and executed
locally by onnxruntime-web on WebGPU where the machine has it and on
single-threaded WebAssembly where it does not.

## Licence, stated plainly

**These weights are AGPL-3.0.** They are an ONNX export of Ultralytics YOLO11n,
and Ultralytics publishes YOLO11 and its pretrained checkpoints under the GNU
Affero General Public License v3.0. Exporting a checkpoint to a different file
format does not change its licence, so this file carries AGPL-3.0 and so does
any use of it.

What that means in practice, for anyone reading this before reusing the file:

- The AGPL's network clause is the part that matters for a hosted page. If you
  serve a modified version of AGPL-3.0 software over a network, you owe your
  users the corresponding source of your modified version.
- The rest of TrafficLens is MIT (see `LICENSE` at the root of the repository).
  The two licences are not the same and this file is the boundary between
  them: the engine, the runtime and the site are MIT; **these weights are not**.
- If you want TrafficLens without the AGPL obligation, swap this file for a
  detector whose licence suits you. Nothing in the runtime is specific to these
  weights beyond the input size and the class ordering described below, both of
  which are parameters rather than assumptions.

Ultralytics also sells a commercial licence for projects that cannot accept the
AGPL. That is a matter between you and them; this note is not legal advice, it
is a pointer to the licence the file actually carries.

## What the graph is

| | |
|---|---|
| Source checkpoint | `yolo11n.pt` (Ultralytics YOLO11 nano) |
| Export | `trafficlens export-model --weights yolo11n.pt --imgsz 480` |
| Opset | ONNX 20, IR version 9 |
| Input | `images`, float32, `[1, 3, 480, 480]`, RGB, CHW, values in `[0, 1]` |
| Output | `output0`, float32, `[1, 84, 4725]` |
| Size | 10 667 823 bytes |
| Weight dtype | float32 throughout; no quantisation operators |

The 4725 output columns are the three detection-head stride grids summed:
`(480/8)^2 + (480/16)^2 + (480/32)^2 = 3600 + 900 + 225`. Rows 0-3 of each
column are `cx, cy, w, h` in letterboxed model pixels; rows 4-83 are the 80 COCO
class scores, already through a sigmoid. There is no separate objectness row and
no NMS baked into the graph — preprocessing and decoding are done by the
runtime, using the same rules the Python engine uses, so both sides can be
compared against each other rather than trusted separately.

## Why float32 and not int8

An int8 dynamic quantisation of this graph was measured against the float32 one
before either was shipped: 40 frames sampled every 15th from the project's
motorway clip, both graphs decoded through identical letterboxing and class-wise
NMS at confidence 0.35 and IoU 0.5.

| | float32 | int8 dynamic |
|---|---|---|
| File size | 10 667 814 B [^1] | 2 976 427 B |
| Detections over 40 frames | 244 | 214 |
| Recall against float32 | — | 0.8238 |
| Mean IoU of matched boxes | — | 0.9301 |

[^1]: 10 667 814 B, nine bytes short of the 10 667 823 B in the table above.
    The two are the same graph measured at different times: the comparison was
    run against an earlier export, and the exporter writes its own version
    string into the graph metadata, so a later toolchain produces a file a few
    bytes longer with identical weights. Nothing about the trade-off below
    turns on nine bytes, but two tables in one document disagreeing is worth a
    line rather than a puzzled reader.

int8 saves 7.69 MB of download and loses 17.6% of detections. The boxes that
survive quantisation are placed well — a mean IoU of 0.93 says the damage is
dropped objects, not displaced ones — and dropped objects are precisely what a
counting product cannot absorb, because a missed detection is a missed count.
So the trade is refused at any download size, and the page pays the 10.7 MB.

## Provenance

Exported from the checkpoint named above with this repository's own CLI, which
runs `ultralytics`' exporter at a fixed input size with dynamic axes disabled.
The export is reproducible from the same checkpoint and the same command; the
file size differs by a few bytes between toolchain versions because the graph
carries the exporter's version string in its metadata.
