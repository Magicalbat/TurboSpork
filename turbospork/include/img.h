/**
 * @file img.h
 * @brief Image operations
 *
 * In each function, an "image" is just a tensor.
 * Each function should work on all color channels (i.e. 2d slices) of each tensor.
 * Values out of bounds in each image are assumed to be zero
 */ 

#ifndef IMG_H
#define IMG_H

#include "base_defs.h"
#include "tensor.h"

/// Sampling methods for image transformations
typedef enum {
    /// Sample nearest pixel
    TS_SAMPLE_NEAREST,
    /// Linear interpolate pixels
    TS_SAMPLE_BILINEAR
} ts_img_sample_type;

// Row major 3d matrix
typedef struct {
    ts_f32 m[9];
} ts_img_mat3;

/**
 * @brief Transforms the input image according to the matrix. In place version
 *
 * Transformations take place about the image's center.
 * Values out of bounds in each image are assumed to be zero
 *
 * @param out Output of transform. The output will be the same size as the input.
 * @param input Input image
 * @param sample_type Sampling type
 * @param mat Transformation matrix
 * 
 * @return true if out is big enough, false otherwise
 */
ts_b32 ts_img_transform_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, const ts_img_mat3* mat);

/**
 * @brief Translates the input image. In place version
 *
 * See `ts_img_transform_ip` for details
 *
 * @param x_off Translation in x-axis
 * @param y_off Translation in y-axis
 */
ts_b32 ts_img_translate_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_off, ts_f32 y_off);

/**
 * @brief Scales the input image. In place version
 *
 * See `ts_img_transform_ip` for details
 *
 * @param x_scale Scale on x-axis
 * @param y_scale Scale on y-axis
 */
ts_b32 ts_img_scale_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_scale, ts_f32 y_scale);

/**
 * @brief Rotates the input image. In place version
 *
 * See `ts_img_transform_ip` for details
 *
 * @param theta Angle to rotate by, in radians
 */
ts_b32 ts_img_rotate_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 theta);

/**
 * @brief Shears the input image. In place version
 *
 * See `ts_img_transform_ip` for details
 *
 * @param x_shear Amount to shear on x axis
 * @param y_shear Amount to shear on y axis
 */
ts_b32 ts_img_shear_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_shear, ts_f32 y_shear);

/**
 * @brief Adds noise to the image. In place version
 *
 * Noise added is between 0 and 1
 *
 * @param input Input image
 * @param sample_type Sampling type
 * @param noise_rate Chance for noise to be added to each 2d pixel of the image.
 *  If noise is added, a different random value is set for each channel of the 2d pixel
 * 
 * @return true if out is big enough, false otherwise
 */
ts_b32 ts_img_add_noise_ip(ts_tensor* out, const ts_tensor* input, ts_f32 noise_rate);

/**
 * @brief Transforms the input image according to the matrix
 *
 * Transformations take place about the image's center.
 * Values out of bounds in each image are assumed to be zero
 *
 * @param arena Arena to create transformed image on
 * @param input Input image
 * @param sample_type Sampling type
 * @param mat Transformation matrix
 * 
 * @return The transformed image if successful, NULL otherwise
 */
ts_tensor* ts_img_transform(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, const ts_img_mat3* mat);

/**
 * @brief Translates the input image
 *
 * See `ts_img_transform` for details
 *
 * @param x_off Translation in x-axis
 * @param y_off Translation in y-axis
 */
ts_tensor* ts_img_translate(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_off, ts_f32 y_off);

/**
 * @brief Scales the input image
 *
 * See `ts_img_transform` for details
 *
 * @param x_scale Scale on x-axis
 * @param y_scale Scale on y-axis
 */
ts_tensor* ts_img_scale(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_scale, ts_f32 y_scale);

/**
 * @brief Rotates the input image
 *
 * See `ts_img_transform` for details
 *
 * @param theta Angle to rotate by, in radians
 */
ts_tensor* ts_img_rotate(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 theta);

/**
 * @brief Shears the input image
 *
 * See `ts_img_transform` for details
 *
 * @param x_shear Amount to shear on x axis
 * @param y_shear Amount to shear on y axis
 */
ts_tensor* ts_img_shear(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_shear, ts_f32 y_shear);

/**
 * @brief Adds noise to the image
 *
 * Noise added is between 0 and 1
 *
 * @param input Input image
 * @param sample_type Sampling type
 * @param noise_rate Chance for noise to be added to each 2d pixel of the image.
 *  If noise is added, a different random value is set for each channel of the 2d pixel
 * 
 * @return Image with noise applied if successful, NULL otherwise
 */
ts_tensor* ts_img_add_noise(mg_arena* arena, const ts_tensor* input, ts_f32 noise_rate);

#endif // IMG_H
