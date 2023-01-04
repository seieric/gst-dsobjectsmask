/**
 * Copyright (c) 2022, seieric
 * This software is based on DeepStream DsExample Plugin by NVIDIA.
 *
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include "gstdsobjectsmask.h"
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC(gst_dsom_debug);
#define GST_CAT_DEFAULT gst_dsom_debug
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_GPU_DEVICE_ID,
  PROP_MIN_CONFIDENCE,
  PROP_CLASS_IDS
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)                                                   \
  ({                                                                                                   \
    int _errtype = 0;                                                                                  \
    do                                                                                                 \
    {                                                                                                  \
      if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) &&      \
          (surface->gpuId != object->gpu_id))                                                          \
      {                                                                                                \
        GST_ELEMENT_ERROR(object, RESOURCE, FAILED,                                                    \
                          ("Input surface gpu-id doesnt match with configured gpu-id for element,"     \
                           " please allocate input using unified memory, or use same gpu-ids"),        \
                          ("surface-gpu-id=%d,%s-gpu-id=%d", surface->gpuId, GST_ELEMENT_NAME(object), \
                           object->gpu_id));                                                           \
        _errtype = 1;                                                                                  \
      }                                                                                                \
    } while (0);                                                                                       \
    _errtype;                                                                                          \
  })

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_GPU_ID 0
#define DEFAULT_MIN_CONFIDENCE 0

#define CHECK_NPP_STATUS(npp_status, error_str)             \
  do                                                        \
  {                                                         \
    if ((npp_status) != NPP_SUCCESS)                        \
    {                                                       \
      g_print("Error: %s in %s at line %d: NPP Error %d\n", \
              error_str, __FILE__, __LINE__, npp_status);   \
      goto error;                                           \
    }                                                       \
  } while (0)

#define CHECK_CUDA_STATUS(cuda_status, error_str)                            \
  do                                                                         \
  {                                                                          \
    if ((cuda_status) != cudaSuccess)                                        \
    {                                                                        \
      g_print("Error: %s in %s at line %d (%s)\n",                           \
              error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
      goto error;                                                            \
    }                                                                        \
  } while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsom_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM,
                                                                              "{ RGBA }")));

static GstStaticPadTemplate gst_dsom_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM,
                                                                              "{ RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsom_parent_class parent_class
G_DEFINE_TYPE(GstDsObjectsMask, gst_dsom, GST_TYPE_BASE_TRANSFORM);

static void gst_dsom_set_property(GObject *object, guint prop_id,
                                  const GValue *value, GParamSpec *pspec);
static void gst_dsom_get_property(GObject *object, guint prop_id,
                                  GValue *value, GParamSpec *pspec);

static gboolean gst_dsom_set_caps(GstBaseTransform *btrans,
                                  GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_dsom_start(GstBaseTransform *btrans);
static gboolean gst_dsom_stop(GstBaseTransform *btrans);

static GstFlowReturn gst_dsom_transform_ip(GstBaseTransform *
                                               btrans,
                                           GstBuffer *inbuf);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsom_class_init(GstDsObjectsMaskClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  /* Indicates we want to use DS buf api */
  g_setenv("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *)klass;
  gstelement_class = (GstElementClass *)klass;
  gstbasetransform_class = (GstBaseTransformClass *)klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_dsom_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_dsom_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_dsom_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_dsom_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_dsom_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR(gst_dsom_transform_ip);

  /* Install properties */
  g_object_class_install_property(gobject_class, PROP_UNIQUE_ID,
                                  g_param_spec_uint("unique-id",
                                                    "Unique ID",
                                                    "Unique ID for the element. Can be used to identify output of the"
                                                    " element",
                                                    0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_GPU_DEVICE_ID,
                                  g_param_spec_uint("gpu-id",
                                                    "Set GPU Device ID",
                                                    "Set GPU Device ID", 0,
                                                    G_MAXUINT, 0,
                                                    GParamFlags(G_PARAM_READWRITE |
                                                                G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property(gobject_class, PROP_MIN_CONFIDENCE,
                                  g_param_spec_double("min-confidence",
                                                      "minimum confidence of objects to be blurred",
                                                      "minimum confidence of objects to be blurred", 0,
                                                      1, DEFAULT_MIN_CONFIDENCE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_CLASS_IDS,
                                  g_param_spec_string("class-ids",
                                                      "class ids",
                                                      "An array of colon-separated class ids for which blur is applied",
                                                      "", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template(gstelement_class,
                                     gst_static_pad_template_get(&gst_dsom_src_template));
  gst_element_class_add_pad_template(gstelement_class,
                                     gst_static_pad_template_get(&gst_dsom_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple(gstelement_class,
                                       "DsObjectsMask plugin",
                                       "DsObjectsMask Plugin",
                                       "Blur objects with cuda",
                                       "seieric");
}

static void
gst_dsom_init(GstDsObjectsMask *dsom)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM(dsom);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

  /* Initialize all property variables to default values */
  dsom->unique_id = DEFAULT_UNIQUE_ID;
  dsom->gpu_id = DEFAULT_GPU_ID;
  dsom->class_ids = new std::set<uint>;

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dsom_set_property(GObject *object, guint prop_id,
                      const GValue *value, GParamSpec *pspec)
{
  GstDsObjectsMask *dsom = GST_DSOM(object);
  switch (prop_id)
  {
  case PROP_UNIQUE_ID:
    dsom->unique_id = g_value_get_uint(value);
    break;
  case PROP_GPU_DEVICE_ID:
    dsom->gpu_id = g_value_get_uint(value);
    break;
  case PROP_MIN_CONFIDENCE:
    dsom->min_confidence = g_value_get_double(value);
    break;
  case PROP_CLASS_IDS:
  {
    std::stringstream str(g_value_get_string(value));
    dsom->class_ids->clear();
    while (str.peek() != EOF)
    {
      gint class_id;
      str >> class_id;
      dsom->class_ids->insert(class_id);
      str.get();
    }
  }
  break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dsom_get_property(GObject *object, guint prop_id,
                      GValue *value, GParamSpec *pspec)
{
  GstDsObjectsMask *dsom = GST_DSOM(object);

  switch (prop_id)
  {
  case PROP_UNIQUE_ID:
    g_value_set_uint(value, dsom->unique_id);
    break;
  case PROP_GPU_DEVICE_ID:
    g_value_set_uint(value, dsom->gpu_id);
    break;
  case PROP_MIN_CONFIDENCE:
    g_value_set_double(value, dsom->min_confidence);
    break;
  case PROP_CLASS_IDS:
  {
    std::stringstream str;
    for (const auto id : *dsom->class_ids)
      str << id << ";";
    g_value_set_string(value, str.str().c_str());
  }
  break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_dsom_start(GstBaseTransform *btrans)
{
  GstDsObjectsMask *dsom = GST_DSOM(btrans);

  GstQuery *queryparams = NULL;
  guint batch_size = 1;
  int val = -1;

  CHECK_CUDA_STATUS(cudaSetDevice(dsom->gpu_id),
                    "Unable to set cuda device");

  cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, dsom->gpu_id);
  dsom->is_integrated = val;

  dsom->batch_size = 1;
  queryparams = gst_nvquery_batch_size_new();
  if (gst_pad_peer_query(GST_BASE_TRANSFORM_SINK_PAD(btrans), queryparams) || gst_pad_peer_query(GST_BASE_TRANSFORM_SRC_PAD(btrans), queryparams))
  {
    if (gst_nvquery_batch_size_parse(queryparams, &batch_size))
    {
      dsom->batch_size = batch_size;
    }
  }
  GST_DEBUG_OBJECT(dsom, "Setting batch-size %d \n",
                   dsom->batch_size);
  gst_query_unref(queryparams);

  CHECK_CUDA_STATUS(cudaStreamCreate(&dsom->cuda_stream),
                    "Could not create cuda stream");

  return TRUE;
error:
  if (dsom->cuda_stream)
  {
    cudaStreamDestroy(dsom->cuda_stream);
    dsom->cuda_stream = NULL;
  }
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_dsom_stop(GstBaseTransform *btrans)
{
  GstDsObjectsMask *dsom = GST_DSOM(btrans);

  if (dsom->cuda_stream)
    cudaStreamDestroy(dsom->cuda_stream);
  dsom->cuda_stream = NULL;

  delete dsom->class_ids;

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dsom_set_caps(GstBaseTransform *btrans, GstCaps *incaps,
                  GstCaps *outcaps)
{
  GstDsObjectsMask *dsom = GST_DSOM(btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps(&dsom->video_info, incaps);

  return TRUE;

error:
  return FALSE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsom_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf)
{
  GstDsObjectsMask *dsom = GST_DSOM(btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  gdouble scale_ratio = 1.0;

  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList *l_frame = NULL;

  dsom->frame_num++;
  CHECK_CUDA_STATUS(cudaSetDevice(dsom->gpu_id),
                    "Unable to set cuda device");

  memset(&in_map_info, 0, sizeof(in_map_info));
  if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ))
  {
    g_print("Error: Failed to map gst buffer\n");
    goto error;
  }

  nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(dsom));
  surface = (NvBufSurface *)in_map_info.data;
  GST_DEBUG_OBJECT(dsom,
                   "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
                   dsom->frame_num, surface);

  if (CHECK_NVDS_MEMORY_AND_GPUID(dsom, surface))
    goto error;

  batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
  if (batch_meta == nullptr)
  {
    GST_ELEMENT_ERROR(dsom, STREAM, FAILED,
                      ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  if (true)
  {
    /* Using object crops as input to the algorithm. The objects are detected by
     * the primary detector */
    NvDsMetaList *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;

    if (!dsom->is_integrated)
    {
      if (!(surface->memType == NVBUF_MEM_CUDA_UNIFIED || surface->memType == NVBUF_MEM_CUDA_PINNED))
      {
        GST_ELEMENT_ERROR(dsom, STREAM, FAILED,
                          ("%s:need NVBUF_MEM_CUDA_UNIFIED or NVBUF_MEM_CUDA_PINNED memory for opencv blurring", __func__), (NULL));
        return GST_FLOW_ERROR;
      }
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
      frame_meta = (NvDsFrameMeta *)(l_frame->data);
      /* Skip all the masking process when no objects are detected. */
      if (frame_meta->num_obj_meta == 0)
        continue;

      if (NvBufSurfaceMapEglImage(surface, frame_meta->batch_id) != 0)
      {
        goto error;
      }
      CUresult status;
      CUeglFrame eglFrame;
      CUgraphicsResource pResource = NULL;
      cudaFree(0);
      status = cuGraphicsEGLRegisterImage(&pResource,
                                          surface->surfaceList[frame_meta->batch_id].mappedAddr.eglImage,
                                          CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
      status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
      status = cuCtxSynchronize();

      cv::cuda::GpuMat in_mat(surface->surfaceList[frame_meta->batch_id].planeParams.height[0],
                              surface->surfaceList[frame_meta->batch_id].planeParams.width[0],
                              CV_8UC4, eglFrame.frame.pPitch[0]);

      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
           l_obj = l_obj->next)
      {
        obj_meta = (NvDsObjectMeta *)(l_obj->data);
        /* skip objects without mask or with confidence less than standard */
        if (obj_meta->mask_params.size <= 0 || obj_meta->confidence < dsom->min_confidence)
          continue;

        /* apply mask only for objects with given class ids */
        auto id_itr = dsom->class_ids->find(obj_meta->class_id);
        if (id_itr == dsom->class_ids->end() || *id_itr != obj_meta->class_id)
          continue;

        /* masking detected area */
#ifdef DEBUG
        g_print("--------マスク処理開始--------\n");
#endif
        // マスク加工の対象範囲
        cv::Rect mask_rect(obj_meta->rect_params.left, obj_meta->rect_params.top,
                           obj_meta->rect_params.width, obj_meta->rect_params.height);
        uint32_t *dst_data;
        if (cudaMalloc((void **)&dst_data, sizeof(uint32_t) * obj_meta->rect_params.width * obj_meta->rect_params.height) != cudaSuccess){
          GST_ELEMENT_ERROR (dsom, STREAM, FAILED,
          ("failed to allocate cuda memory"), (NULL));
          if (NvBufSurfaceUnMapEglImage (surface, frame_meta->batch_id) != 0){
            GST_ELEMENT_ERROR (dsom, STREAM, FAILED,
            ("%s:buffer unmap failed", __func__), (NULL));
          }
          return GST_FLOW_ERROR;
        }
#ifdef DEBUG
        g_print("マスクをARGB32形式に変更します。\n");
#endif
        // refer to https://docs.nvidia.com/metropolis/deepstream/sdk-api/nvds__mask__utils_8h.html
        if (!nvds_mask_utils_resize_to_binary_argb32(obj_meta->mask_params.data, dst_data,
                                                obj_meta->mask_params.width, obj_meta->mask_params.height,
                                                obj_meta->rect_params.width, obj_meta->rect_params.height,
                                                1, obj_meta->mask_params.threshold,
                                                16777215, 1,
                                                dsom->cuda_stream))
        {
          GST_ELEMENT_ERROR (dsom, STREAM, FAILED,
          ("failed to convert mask"), (NULL));
          if (NvBufSurfaceUnMapEglImage (surface, frame_meta->batch_id) != 0){
            GST_ELEMENT_ERROR (dsom, STREAM, FAILED,
            ("%s:buffer unmap failed", __func__), (NULL));
          }
          return GST_FLOW_ERROR;
        }
#ifdef DEBUG
        g_print("OpenCVでマスクを初期化します。\n");
#endif       
        // mask image（ARGB）
        // CV_32SC4、CV32SC1は不可
        cv::cuda::GpuMat mask(obj_meta->rect_params.height, obj_meta->rect_params.width, CV_8UC4, dst_data);
        // グレースケールに変換
        cv::cuda::GpuMat gray_mask;
        cv::cuda::cvtColor(mask, gray_mask, cv::COLOR_RGBA2GRAY);
        // 白黒に変換
        cv::cuda::threshold(gray_mask, gray_mask, 127, 255, cv::THRESH_BINARY);
        // カラーに戻す（1 channel to 4 channels）
        cv::cuda::cvtColor(gray_mask, mask, cv::COLOR_GRAY2RGBA);
        gray_mask.release();
        // 検出領域を白塗りにする
        cv::cuda::bitwise_or(in_mat(mask_rect), mask, in_mat(mask_rect));
#ifdef DEBUG
        g_print("%d\n", obj_meta->mask_params.size);
        g_print("%f\n", obj_meta->mask_params.threshold);
        g_print("%ux%u\n", obj_meta->mask_params.width, obj_meta->mask_params.height);
        g_print("%fx%f\n", obj_meta->rect_params.width, obj_meta->rect_params.height);
        g_print("%fx%f\n", obj_meta->rect_params.top, obj_meta->rect_params.left);
        g_print("%dx%d\n", mask.size().width, mask.size().height);
        g_print("%dx%d\n", in_mat(mask_rect).size().width, in_mat(mask_rect).size().height);
        g_print("OpenCVでマスクを開放します。\n");
#endif
        mask.release();
#ifdef DEBUG
        g_print("OpenCVでマスクを開放しました。cudaメモリを開放します。\n");
#endif
        if (cudaFree(dst_data) != cudaSuccess) {
          GST_ELEMENT_ERROR (dsom, STREAM, FAILED,
          ("failed to free cuda memory"), (NULL));
          if (NvBufSurfaceUnMapEglImage (surface, frame_meta->batch_id) != 0){
            GST_ELEMENT_ERROR (dsom, STREAM, FAILED,
            ("%s:buffer unmap failed", __func__), (NULL));
          }
          return GST_FLOW_ERROR;
        }
#ifdef DEBUG
        g_print("--------マスク処理終了--------\n");
#endif
      }

      status = cuCtxSynchronize();
      status = cuGraphicsUnregisterResource(pResource);
      // Destroy the EGLImage
      NvBufSurfaceUnMapEglImage(surface, frame_meta->batch_id);
    }
  }
  flow_ret = GST_FLOW_OK;

error:

  nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(dsom));
  gst_buffer_unmap(inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsom_plugin_init(GstPlugin *plugin)
{
  GST_DEBUG_CATEGORY_INIT(gst_dsom_debug, "dsobjectsmask", 0,
                          "dsobjectsmask plugin");

  return gst_element_register(plugin, "dsobjectsmask", GST_RANK_PRIMARY,
                              GST_TYPE_DSOM);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_dsobjectsmask,
                  DESCRIPTION, dsom_plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE, URL)