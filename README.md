# Gst-dsobjectsmask

This plugin masks objects detected by NVIDIA nvinfer plugin. Fast and smooth since all the masking processes are done with GPU.

Must be a better masking solution than masking with nvdsosd.

**Note: This plugin is tested with PeopleSegNet, which is an instance segmentation model. Masking process refers to a object's mask_params of NvDsObjectMeta. Other models (e.g. semantic segmentation models) may not attach mask_params and this wouldn't work as expected.**

**Note: For Jetson only, not works with dGPU.**

![](https://raw.githubusercontent.com/seieric/gst-dsobjectsmask/main/gst-dsobjectsmask.png "")

## Features
- Mask objects with cuda
- Specify class ids for which blur should be applied
- Fast and smooth processing

## Gst Properties
| Property | Meaning | Type and Range |
| -------- | ------- | -------------- |
| min-confidence | Minimum confidence of objects to be masked | Double, 0 to 1
| class-ids | Class ids of objects for which masking should be applied | Semicolon delimited integer array |

## Depedencies
- DeepStream 6.1
- OpenCV4 with CUDA support
## Download and Installation
If your environment satisfies the requirements, just run following commands.
```bash
git clone https://github.com/seieric/gst-dsobjectsmask.git
cd gst-dsobjectsmask
sudo make -j$(nproc) install
```

## Example usage
This is a brief instruction to test this plugin after installtion.
1. Download [NVIDIA-AI-IOT/deepstream_tao_apps](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps) to ```/opt/nvidia/deepstream/deepstream-6.1/sources/apps```.
2. Download required models for deepstream_tao_apps. Refer to [the documentation](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps#2-download-models).
3. Run the sample script ```samples/peoplesegnet_test.sh```.
4. You will get the ```peoplesegnet_test.mp4``` in your curennt directory.