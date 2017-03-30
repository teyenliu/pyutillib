import pandas as pd
import tensorflow as tf

CATEGORICAL_COLUMNS = ["A", "B", "C", "D", "E", "F", "I", "J"]
CONTINUOUS_COLUMNS = ["G", "H", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]

SURVIVED_COLUMN = "T"

def build_estimator(model_dir):
  """Build an estimator."""
  A = tf.contrib.layers.sparse_column_with_hash_bucket("A", hash_bucket_size=1000)
  B = tf.contrib.layers.sparse_column_with_hash_bucket("B", hash_bucket_size=1000)
  C = tf.contrib.layers.sparse_column_with_hash_bucket("C", hash_bucket_size=1000)
  D = tf.contrib.layers.sparse_column_with_hash_bucket("D", hash_bucket_size=1000)
  E = tf.contrib.layers.sparse_column_with_hash_bucket("E", hash_bucket_size=1000)
  F = tf.contrib.layers.sparse_column_with_hash_bucket("F", hash_bucket_size=1000)
  I = tf.contrib.layers.sparse_column_with_hash_bucket("I", hash_bucket_size=1000)
  J = tf.contrib.layers.sparse_column_with_hash_bucket("J", hash_bucket_size=1000)


  # Continuous columns
  G = tf.contrib.layers.real_valued_column("G")
  H = tf.contrib.layers.real_valued_column("H")
  K = tf.contrib.layers.real_valued_column("K")
  L = tf.contrib.layers.real_valued_column("L")
  M = tf.contrib.layers.real_valued_column("M")
  N = tf.contrib.layers.real_valued_column("N")
  O = tf.contrib.layers.real_valued_column("O")
  P = tf.contrib.layers.real_valued_column("P")
  Q = tf.contrib.layers.real_valued_column("Q")
  R = tf.contrib.layers.real_valued_column("R")
  S = tf.contrib.layers.real_valued_column("S")

  # Transformations.
  #age_buckets = tf.contrib.layers.bucketized_column(age,
  #                                                  boundaries=[
  #                                                      5, 18, 25, 30, 35, 40,
  #                                                      45, 50, 55, 65
  #                                                  ])

  # Wide columns and deep columns.
  wide_columns = [A, B, C, D, E, F, I, J]
#                  tf.contrib.layers.crossed_column([A, B], hash_bucket_size=int(1e6)),
#                  tf.contrib.layers.crossed_column([C, D], hash_bucket_size=int(1e6)),
#                  tf.contrib.layers.crossed_column([E, F], hash_bucket_size=int(1e6)),
#                  tf.contrib.layers.crossed_column([I, J], hash_bucket_size=int(1e6))]
  deep_columns = [
      #tf.contrib.layers.embedding_column(A, dimension=8),
      #tf.contrib.layers.embedding_column(B, dimension=8),
      #tf.contrib.layers.embedding_column(C, dimension=8),
      #tf.contrib.layers.embedding_column(D, dimension=8),
      #tf.contrib.layers.embedding_column(E, dimension=8),
      #tf.contrib.layers.embedding_column(F, dimension=8),
      #tf.contrib.layers.embedding_column(I, dimension=8),
      #tf.contrib.layers.embedding_column(J, dimension=8),
      G,
      H,
      K,
      L,
      M,
      N,
      O,
      P,
      Q,
      R,
      S
  ]

  return tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 100])

def input_fn(df, train=False):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
    indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  if train:
    label = tf.constant(df[SURVIVED_COLUMN].values)
      # Returns the feature columns and the label.
    return feature_cols, label
  else:
    return feature_cols


def train_and_eval():
  """Train and evaluate the model."""
  df_train = pd.read_csv(
      "./train.csv",
      skipinitialspace=True)
  df_test = pd.read_csv(
      "./test.csv",
      skipinitialspace=True)

  model_dir = "./models"
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  m.fit(input_fn=lambda: input_fn(df_train, True), steps=1000)
  print m.predict(input_fn=lambda: input_fn(df_test))
  results = m.evaluate(input_fn=lambda: input_fn(df_train, True), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
