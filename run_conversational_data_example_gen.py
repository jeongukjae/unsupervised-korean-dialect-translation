"""
한국어 대화 example gen script
"""
import glob
import os
import zipfile
import openpyxl

import tensorflow as tf
from absl import app, flags, logging
from tqdm import tqdm

DIRECTORY_NAME = "한국어 대화"
TFRECORD_NAME = "서울"

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

    dataset_path = os.path.join(FLAGS.base_path, DIRECTORY_NAME)
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Cannot find dataset directory {dataset_path}")

    file = glob.glob(os.path.join(dataset_path, "*.zip"))
    # zip 파일은 하나 있어요.
    if len(file) != 1:
        raise ValueError("Malformed dataset error, there are multiple zip files or a zip file does not exist")

    unzip_path = os.path.join(FLAGS.temp_path, DIRECTORY_NAME)
    logging.info(f"Unzipping {file[0]} into {unzip_path}")
    with zipfile.ZipFile(file[0], "r") as zf:
        zf.extractall(unzip_path)

    xlsx_files = glob.glob(os.path.join(unzip_path, "*.xlsx"))
    if len(xlsx_files) == 0:
        raise ValueError("Malformed dataset error, cannot find any xlsx files")
    logging.info(f"Found {len(xlsx_files)} files")

    sentence_count = 0
    with tf.io.TFRecordWriter(os.path.join(FLAGS.output_path, f"dialect-{TFRECORD_NAME}.tfrecord")) as tfrecord_writer:
        for xlsx_file in tqdm(xlsx_files, desc="Processing xlsx files.."):
            logging.info(f"Processing {xlsx_file}")
            workbook = openpyxl.load_workbook(xlsx_file)
            worksheet = workbook.active

            rows = list(worksheet.rows)
            sentences = list({row[1].value for row in rows[1:]})
            logging.info(f"Found {len(sentences)} rows.")
            logging.info(f"sentences[0]: {sentences[0]}")
            workbook.close()

            for sentence in sentences:
                if type(sentence) != str:
                    # skip integer cells
                    continue

                sentence_count += 1
                sentence_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sentence.encode("utf-8")]))
                language_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[TFRECORD_NAME.encode("utf-8")]))
                features = tf.train.Features(feature={"sentence": sentence_feature, "lang": language_feature})
                tfrecord_writer.write(tf.train.Example(features=features).SerializeToString())

    logging.info(f"# sentences: {sentence_count}")
    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
