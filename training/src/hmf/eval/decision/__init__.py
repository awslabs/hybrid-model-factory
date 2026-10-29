"""
Evaluation of decision models (stage `decision`), which calibrates them and evaluates them on decision data files.
The definitions are the same as in Strands Decider's evaluation, so that the numbers can be compared
with the ones it publishes.

    - predictor.py: DecisionPredictor, a trained decision model's probabilities, raw or calibrated.
    - calibrate.py: one temperature per question kind, fitted to minimize the expected calibration
      error, saved in the adapter's folder (the training output_dir).
    - evaluate.py: accuracy, ECE, NLL, and more, overall and by kind, task, and confidence band.
    - server.py: a decision server (JevBench's typesafe protocol), for running JevBench.
    - jevbench.py: the per-task comparison of two JevBench runs (exact McNemar test).
    - suite.py: evaluation suites, which run calibration, benchmarks, and JevBench from one YAML file.
    - cli.py: the hmf decision-eval and hmf decision-serve commands.
    - metrics.py: the confidence of an answer, and the metrics.
    - data.py: reading decision data files, and the split of a file into a calibration half and a
      test half.

See README.md in this folder for the benchmarks and the procedure.
"""
