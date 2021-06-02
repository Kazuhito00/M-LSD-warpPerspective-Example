#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import CvFpsCalc
from mlsd.utils import pred_lines, pred_squares


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--crop_width", type=int, default=224)
    parser.add_argument("--crop_height", type=int, default=224)

    parser.add_argument(
        "--model",
        type=str,
        default='mlsd/tflite_models/M-LSD_320_tiny_fp32.tflite')
    parser.add_argument("--model_shape", type=int, default=320)
    parser.add_argument("--top_n", type=int, default=1)

    parser.add_argument("--score", type=float, default=0.1)
    parser.add_argument("--outside_ratio", type=float, default=0.1)
    parser.add_argument("--inside_ratio", type=float, default=0.5)
    parser.add_argument("--w_overlap", type=float, default=0.0)
    parser.add_argument("--w_degree", type=float, default=1.14)
    parser.add_argument("--w_length", type=float, default=0.03)
    parser.add_argument("--w_area", type=float, default=1.84)
    parser.add_argument("--w_center", type=float, default=1.46)

    args = parser.parse_args()

    return args


def get_params(args):
    params = {
        'score': args.score,
        'outside_ratio': args.outside_ratio,
        'inside_ratio': args.inside_ratio,
        'w_overlap': args.w_overlap,
        'w_degree': args.w_degree,
        'w_length': args.w_length,
        'w_area': args.w_area,
        'w_center': args.w_center,
    }
    return params


def extract_square_image(image, square, crop_width, crop_height):
    extract_image = None

    # 射影変換
    pts1 = np.float32([
        square[0],
        square[1],
        square[2],
        square[3],
    ])
    pts2 = np.float32([
        [0, 0],
        [crop_width, 0],
        [crop_width, crop_height],
        [0, crop_height],
    ])
    M = cv.getPerspectiveTransform(pts1, pts2)
    extract_image = cv.warpPerspective(image, M, (crop_width, crop_height))

    return extract_image


def main():
    # コマンドライン引数
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    filepath = args.file

    crop_width = args.crop_width
    crop_height = args.crop_height

    model = args.model
    model_shape = args.model_shape
    model_shapes = [model_shape, model_shape]

    top_n = args.top_n

    # カメラ準備
    cap = None
    if filepath is None:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        cap = cv.VideoCapture(filepath)

    # M-LSDモデルロード
    interpreter = tf.lite.Interpreter(model_path=model, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    params = get_params(args)

    # FPS計測モジュール
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    square_count = 0
    prev_square_count = 0
    while True:
        # キー入力(ESC:プログラム終了)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # FPS計測
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ
        ret, frame = cap.read()
        if not ret:
            print('Error : cap.read()')
        if filepath is None:
            resize_frame = cv.resize(frame, (int(cap_width), int(cap_height)))
        else:
            resize_frame = copy.deepcopy(frame)

        # M-LSD推論
        lines, squares, score_array, inter_points = pred_squares(
            resize_frame,
            interpreter,
            input_details,
            output_details,
            model_shapes,
            params,
        )

        # スコア降順にインデックスを並び替え
        # sorted_score_array = []
        sorted_squares = []
        if (len(score_array) > 0):
            score_sort_index = np.argsort(score_array)[::-1]
            # sorted_score_array = score_array[score_sort_index]
            sorted_squares = squares[score_sort_index]

        # 射影変換
        extract_images = []
        for index, square in enumerate(sorted_squares):
            if (index < top_n):
                extract_image = extract_square_image(resize_frame, square,
                                                     crop_width, crop_height)
                extract_images.append(extract_image)

                square_count = index + 1
            else:
                break

        # デバッグ情報描画
        for index, square in enumerate(sorted_squares):
            if (index < top_n):
                cv.polylines(resize_frame, [square.reshape([-1, 1, 2])], True,
                             [255, 0, 0], 2)
                for pt in square:
                    cv.circle(resize_frame, (int(pt[0]), int(pt[1])), 4,
                              [255, 0, 0], -1)
            else:
                break

        cv.putText(resize_frame, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)

        # 描画更新
        cv.imshow('M-LSD-warpPerspective', resize_frame)
        for index, extract_image in enumerate(extract_images):
            cv.imshow('SQUARE:' + str(index), extract_image)

        # 不要ウィンドウ削除
        if prev_square_count > len(extract_images):
            for index in range(len(extract_images), prev_square_count):
                cv.destroyWindow('SQUARE:' + str(index))
        prev_square_count = square_count

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
