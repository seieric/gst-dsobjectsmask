#!/bin/bash
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream-6.1/samples/streams/sample_walk.mov ! \
qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux name=m sync-inputs=0 batch-size=1 width=1920 height=1080 ! \
queue ! nvvideoconvert ! nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/sources/apps/deepstream_tao_apps/configs/peopleSegNet_tao/pgie_peopleSegNetv2_tao_config.txt ! \
queue ! dsobjectsmask class-ids="0;1" min-confidence=0 ! queue ! nvvideoconvert \
! nvv4l2h264enc ! h264parse ! qtmux ! queue ! filesink location=./peoplesegnet_test.mp4