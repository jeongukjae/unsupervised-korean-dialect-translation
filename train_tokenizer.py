import os

import tensorflow as tf
from absl import app, flags, logging
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", default="./data/", help="dataset base path")
flags.DEFINE_string("output_path", default="./vocab.txt", help="bert tokenizer path")
flags.DEFINE_integer("vocab_size", default=8000, help="vocab size")


def main(argv):
    logging.info(f"dataset path: {FLAGS.data_path}")
    logging.info(f"output path: {FLAGS.output_path}")
    logging.info(f"vocab size: {FLAGS.vocab_size}")

    dataset = (
        tf.data.Dataset.list_files(os.path.join(FLAGS.data_path, "*.tfrecord"))
        .interleave(
            lambda filename: tf.data.TFRecordDataset([filename], num_parallel_reads=tf.data.AUTOTUNE),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(1000)
        .map(lambda x: tf.io.parse_example(x, {"sentence": tf.io.FixedLenFeature([], dtype=tf.string)})["sentence"])
        .cache()
    )
    length = len([1 for _ in dataset])
    dataset = dataset.shuffle(length).take(500).cache()
    # # sentences: 500,000

    logging.info(f"dataset element spec: {dataset.element_spec}")

    reserved_tokens = ["[PAD]", "[UNK]", "[END]", "[서울]", "[강원]", "[경상]", "[전라]", "[제주]", "[충청]"]
    vocab = bert_vocab.bert_vocab_from_dataset(
        dataset,
        vocab_size=FLAGS.vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=dict(lower_case=True, normalization_form="NFD"),
        learn_params={},
    )
    logging.info(f"vocab[:10]: {vocab[:10]}")
    with open(FLAGS.output_path, "w") as f:
        for token in vocab:
            print(token, file=f)

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
