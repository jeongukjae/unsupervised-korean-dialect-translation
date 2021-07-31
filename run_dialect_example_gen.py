"""
각 방언 발화 example gen script

base path를 한곳에 두고 다운로드 받아주세요.
"""
import glob
import json
import os
import zipfile

import tensorflow as tf
from absl import app, flags, logging
from tqdm import tqdm

DIRECTORY_NAMES = [
    "한국어 방언 발화 데이터(강원도)",
    "한국어 방언 발화 데이터(경상도)",
    "한국어 방언 발화 데이터(전라도)",
    "한국어 방언 발화 데이터(제주도)",
    "한국어 방언 발화 데이터(충청도)",
]
TFRECORD_NAMES = ["강원", "경상", "전라", "제주", "충청"]

FLAGS = flags.FLAGS
flags.DEFINE_string("base_path", default=os.path.expanduser("~/Downloads"), help="방언 데이터의 기본 경로")
flags.DEFINE_string("temp_path", default="./tmp", help="임시로 사용할 경로")
flags.DEFINE_string("output_path", default="./data", help="TFRecord가 저장될 경로")


def main(argv):
    logging.info(f"base path: {FLAGS.base_path}")
    logging.info(f"temp path: {FLAGS.temp_path}")
    logging.info(f"output path: {FLAGS.output_path}")

    if not os.path.isdir(FLAGS.temp_path):
        os.makedirs(FLAGS.temp_path)
    if not os.path.isdir(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    for dirname, tfrecord_name in zip(DIRECTORY_NAMES, TFRECORD_NAMES):
        logging.info(f"Processing '{dirname}'")
        dataset_path = os.path.join(FLAGS.base_path, dirname)
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Cannot find dataset directory {dataset_path}")

        file = glob.glob(os.path.join(dataset_path, "Training", "*.zip"))
        # zip 파일은 하나 있어요.
        if len(file) != 1:
            raise ValueError("Malformed dataset error, there are multiple zip files or a zip file does not exist")

        unzip_path = os.path.join(FLAGS.temp_path, dirname)
        logging.info(f"Unzipping {file[0]} into {unzip_path}")
        with zipfile.ZipFile(file[0], "r") as zf:
            zf.extractall(unzip_path)

        json_files = glob.glob(os.path.join(unzip_path, "**", "*.json"), recursive=True)
        if len(json_files) == 0:
            raise ValueError("Malformed dataset error, cannot find any json files")
        logging.info(f"Found {len(json_files)} files")

        sentence_count = 0
        with tf.io.TFRecordWriter(os.path.join(FLAGS.output_path, f"dialect-{tfrecord_name}.tfrecord")) as tfrecord_writer:
            for json_file in tqdm(json_files, desc="Processing json files.."):
                try:
                    with open(json_file, "r") as f:
                        dialect_file_json = json.load(f)
                        dialects = [utterance["dialect_form"] for utterance in dialect_file_json["utterance"]]
                        for dialect in dialects:
                            sentence_count += 1
                            sentence_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[dialect.encode("utf-8")]))
                            language_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tfrecord_name.encode("utf-8")]))
                            features = tf.train.Features(feature={"sentence": sentence_feature, "lang": language_feature})
                            tfrecord_writer.write(tf.train.Example(features=features).SerializeToString())
                except Exception as e:
                    logging.error(f"Got error while processing file {json_file}. Skipping.\nerror msg: {e}")

        logging.info(f"# sentences: {sentence_count}")
        logging.info(f"Done '{dirname}'.")


if __name__ == "__main__":
    app.run(main)
