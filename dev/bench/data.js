window.BENCHMARK_DATA = {
  "lastUpdate": 1764908139377,
  "repoUrl": "https://github.com/bogsdarking/planet_ruler",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "100499183+bogsdarking@users.noreply.github.com",
            "name": "Brandon Anderson",
            "username": "bogsdarking"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7bbc885122258bb79dfcf8bcdbbdccaa9c154ff3",
          "message": "Merge pull request #22 from bogsdarking/dev\n\nUse dedicated benchmarks branch for performance tracking",
          "timestamp": "2025-10-14T13:38:43-04:00",
          "tree_id": "7ca16380b89b4cb47069d536ba258b615d7acc68",
          "url": "https://github.com/bogsdarking/planet_ruler/commit/7bbc885122258bb79dfcf8bcdbbdccaa9c154ff3"
        },
        "date": 1760464498549,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_benchmark",
            "value": 635321.4127361035,
            "unit": "iter/sec",
            "range": "stddev: 3.3314936257763905e-7",
            "extra": "mean: 1.5740064476866211 usec\nrounds: 30554"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_camera_angle_benchmark",
            "value": 1005506.6390435298,
            "unit": "iter/sec",
            "range": "stddev: 2.522125494198683e-7",
            "extra": "mean: 994.5235179662584 nsec\nrounds: 77919"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_vectorized_benchmark",
            "value": 866.9931353433363,
            "unit": "iter/sec",
            "range": "stddev: 0.00001824358592002914",
            "extra": "mean: 1.1534116698674806 msec\nrounds: 833"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_small",
            "value": 3526.3363698911776,
            "unit": "iter/sec",
            "range": "stddev: 0.000014325205479641653",
            "extra": "mean: 283.5804345093885 usec\nrounds: 1275"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_large",
            "value": 1624.3572935059942,
            "unit": "iter/sec",
            "range": "stddev: 0.00002219483536530983",
            "extra": "mean: 615.6281034953901 usec\nrounds: 1459"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_coordinate_transforms_benchmark",
            "value": 8169.153783224292,
            "unit": "iter/sec",
            "range": "stddev: 0.000005642469066894187",
            "extra": "mean: 122.4116997348664 usec\nrounds: 5282"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_packing_benchmark",
            "value": 1368024.334443181,
            "unit": "iter/sec",
            "range": "stddev: 5.687367416565217e-8",
            "extra": "mean: 730.9811491087415 nsec\nrounds: 67400"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_unpacking_benchmark",
            "value": 648020.7565659155,
            "unit": "iter/sec",
            "range": "stddev: 4.2501327882226983e-7",
            "extra": "mean: 1.5431604464328323 usec\nrounds: 119261"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_evaluation_benchmark",
            "value": 41760.53846015337,
            "unit": "iter/sec",
            "range": "stddev: 0.000009937771293900765",
            "extra": "mean: 23.946051389020504 usec\nrounds: 23721"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_different_losses_benchmark",
            "value": 4298.607774238873,
            "unit": "iter/sec",
            "range": "stddev: 0.000011143357543979548",
            "extra": "mean: 232.63346006883907 usec\nrounds: 2617"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_small",
            "value": 5.963290771550101,
            "unit": "iter/sec",
            "range": "stddev: 0.001398511810069471",
            "extra": "mean: 167.69264459999818 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_large",
            "value": 1.4115674100758242,
            "unit": "iter/sec",
            "range": "stddev: 0.005749695901358478",
            "extra": "mean: 708.4323376000043 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_realistic",
            "value": 3.666193821029391,
            "unit": "iter/sec",
            "range": "stddev: 0.001128538408327367",
            "extra": "mean: 272.76244760000736 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_rolling_median",
            "value": 1338.5408863142668,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325635344249727",
            "extra": "mean: 747.082147601442 usec\nrounds: 813"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_savgol",
            "value": 2849.2566589808644,
            "unit": "iter/sec",
            "range": "stddev: 0.000016159978162286913",
            "extra": "mean: 350.96873314237854 usec\nrounds: 1750"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_methods_comparison_benchmark",
            "value": 547.3163527910675,
            "unit": "iter/sec",
            "range": "stddev: 0.00012011697377058022",
            "extra": "mean: 1.8270968789813227 msec\nrounds: 471"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_planet_observation_initialization_benchmark",
            "value": 138908.07637313282,
            "unit": "iter/sec",
            "range": "stddev: 0.000011699076418030687",
            "extra": "mean: 7.199005458212629 usec\nrounds: 30047"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_detect_limb_gradient_break_benchmark",
            "value": 3.6730197456729443,
            "unit": "iter/sec",
            "range": "stddev: 0.0015804277130094392",
            "extra": "mean: 272.25554700000316 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_complete_geometry_pipeline_benchmark",
            "value": 2336.3544011909703,
            "unit": "iter/sec",
            "range": "stddev: 0.000017249331989910633",
            "extra": "mean: 428.0172560679339 usec\nrounds: 1648"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_image_processing_pipeline_benchmark",
            "value": 3.9168858843898415,
            "unit": "iter/sec",
            "range": "stddev: 0.0015589377527755568",
            "extra": "mean: 255.30485940000176 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_parameter_optimization_simulation_benchmark",
            "value": 536.4047717987911,
            "unit": "iter/sec",
            "range": "stddev: 0.00004067383751852705",
            "extra": "mean: 1.864263803333775 msec\nrounds: 300"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "100499183+bogsdarking@users.noreply.github.com",
            "name": "Brandon Anderson",
            "username": "bogsdarking"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c24428aa53d5dbcf0aaa1e042749ed5cd2f89b50",
          "message": "v1.3.0",
          "timestamp": "2025-10-19T19:41:43-04:00",
          "tree_id": "ae71f91dbef27ee616f9006526202811d024e4e5",
          "url": "https://github.com/bogsdarking/planet_ruler/commit/c24428aa53d5dbcf0aaa1e042749ed5cd2f89b50"
        },
        "date": 1760917792286,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_benchmark",
            "value": 627867.0758335473,
            "unit": "iter/sec",
            "range": "stddev: 3.591500216819543e-7",
            "extra": "mean: 1.592693801745241 usec\nrounds: 31251"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_camera_angle_benchmark",
            "value": 980261.4359603316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020223397679700734",
            "extra": "mean: 1.020136020163163 usec\nrounds: 70938"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_vectorized_benchmark",
            "value": 866.9877083900221,
            "unit": "iter/sec",
            "range": "stddev: 0.000015228135274246422",
            "extra": "mean: 1.1534188897060362 msec\nrounds: 816"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_small",
            "value": 3559.5391884144683,
            "unit": "iter/sec",
            "range": "stddev: 0.000013718758314686067",
            "extra": "mean: 280.93524107131174 usec\nrounds: 1344"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_large",
            "value": 1614.4314412664194,
            "unit": "iter/sec",
            "range": "stddev: 0.00002424870793492206",
            "extra": "mean: 619.4131100516497 usec\nrounds: 1363"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_coordinate_transforms_benchmark",
            "value": 8074.729009823543,
            "unit": "iter/sec",
            "range": "stddev: 0.000005490643974589688",
            "extra": "mean: 123.84316535990513 usec\nrounds: 5739"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_packing_benchmark",
            "value": 1119691.9711874088,
            "unit": "iter/sec",
            "range": "stddev: 2.986794643836807e-7",
            "extra": "mean: 893.1027690941838 nsec\nrounds: 109566"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_unpacking_benchmark",
            "value": 626889.7989854717,
            "unit": "iter/sec",
            "range": "stddev: 6.632047320135981e-7",
            "extra": "mean: 1.5951766987090106 usec\nrounds: 165536"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_evaluation_benchmark",
            "value": 40848.64343276846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016396128698850102",
            "extra": "mean: 24.480617126144455 usec\nrounds: 24115"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_different_losses_benchmark",
            "value": 4416.140596996297,
            "unit": "iter/sec",
            "range": "stddev: 0.000011597237478707202",
            "extra": "mean: 226.4420658799144 usec\nrounds: 2459"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_small",
            "value": 6.004288815466564,
            "unit": "iter/sec",
            "range": "stddev: 0.0010203929207559298",
            "extra": "mean: 166.54761800000037 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_large",
            "value": 1.3992500357307591,
            "unit": "iter/sec",
            "range": "stddev: 0.017562459679262905",
            "extra": "mean: 714.6685542 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_realistic",
            "value": 3.616238854231087,
            "unit": "iter/sec",
            "range": "stddev: 0.0003050122865472958",
            "extra": "mean: 276.530406400002 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_rolling_median",
            "value": 1324.0797315779412,
            "unit": "iter/sec",
            "range": "stddev: 0.00002109192166361262",
            "extra": "mean: 755.2415282486601 usec\nrounds: 354"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_savgol",
            "value": 2775.661673767183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000401645991420484",
            "extra": "mean: 360.274456159774 usec\nrounds: 1802"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_methods_comparison_benchmark",
            "value": 556.7287204479587,
            "unit": "iter/sec",
            "range": "stddev: 0.00002370521008228697",
            "extra": "mean: 1.7962069555085525 msec\nrounds: 472"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_planet_observation_initialization_benchmark",
            "value": 140222.1209514774,
            "unit": "iter/sec",
            "range": "stddev: 0.000016123008767356515",
            "extra": "mean: 7.1315423929869155 usec\nrounds: 28625"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_detect_limb_gradient_break_benchmark",
            "value": 3.6624207505071165,
            "unit": "iter/sec",
            "range": "stddev: 0.0008564851779403319",
            "extra": "mean: 273.04345080000303 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_complete_geometry_pipeline_benchmark",
            "value": 2342.540592744678,
            "unit": "iter/sec",
            "range": "stddev: 0.000015873769260852973",
            "extra": "mean: 426.88694620584255 usec\nrounds: 1766"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_image_processing_pipeline_benchmark",
            "value": 3.9255359619039543,
            "unit": "iter/sec",
            "range": "stddev: 0.0005694087808765472",
            "extra": "mean: 254.74228480000534 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_parameter_optimization_simulation_benchmark",
            "value": 532.8852388430179,
            "unit": "iter/sec",
            "range": "stddev: 0.00003420113907580671",
            "extra": "mean: 1.8765766568636162 msec\nrounds: 510"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "100499183+bogsdarking@users.noreply.github.com",
            "name": "Brandon Anderson",
            "username": "bogsdarking"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "43a859f1d3a92a4082061bace1b2faacc018f448",
          "message": "v1.4.0\n\nGradient field fit method.",
          "timestamp": "2025-11-03T11:36:31-05:00",
          "tree_id": "3d176a56c6eab11e742aa3bb3e867e6502d2fb5c",
          "url": "https://github.com/bogsdarking/planet_ruler/commit/43a859f1d3a92a4082061bace1b2faacc018f448"
        },
        "date": 1762188248339,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_benchmark",
            "value": 626784.9147159806,
            "unit": "iter/sec",
            "range": "stddev: 3.8660576197786764e-7",
            "extra": "mean: 1.5954436306960849 usec\nrounds: 31134"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_camera_angle_benchmark",
            "value": 1006596.8639706746,
            "unit": "iter/sec",
            "range": "stddev: 2.8883066556646414e-7",
            "extra": "mean: 993.4463694386526 nsec\nrounds: 75787"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_vectorized_benchmark",
            "value": 874.9073044397468,
            "unit": "iter/sec",
            "range": "stddev: 0.000017657806464527626",
            "extra": "mean: 1.1429782274367424 msec\nrounds: 831"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_small",
            "value": 6308.842800981756,
            "unit": "iter/sec",
            "range": "stddev: 0.000009309199527098059",
            "extra": "mean: 158.50767431459602 usec\nrounds: 1787"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_large",
            "value": 3087.45333107841,
            "unit": "iter/sec",
            "range": "stddev: 0.000011487521669226944",
            "extra": "mean: 323.89153543924573 usec\nrounds: 2469"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_coordinate_transforms_benchmark",
            "value": 8135.846673535986,
            "unit": "iter/sec",
            "range": "stddev: 0.000009961953802144785",
            "extra": "mean: 122.9128374865725 usec\nrounds: 5538"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_packing_benchmark",
            "value": 1356970.1864243946,
            "unit": "iter/sec",
            "range": "stddev: 7.137507280301984e-8",
            "extra": "mean: 736.9358663914218 nsec\nrounds: 65369"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_unpacking_benchmark",
            "value": 638323.6235276802,
            "unit": "iter/sec",
            "range": "stddev: 4.5741875950882585e-7",
            "extra": "mean: 1.5666034643579756 usec\nrounds: 131857"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_evaluation_benchmark",
            "value": 41816.09241801619,
            "unit": "iter/sec",
            "range": "stddev: 0.000001752010142427055",
            "extra": "mean: 23.914238327279868 usec\nrounds: 25444"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_different_losses_benchmark",
            "value": 4407.668387535394,
            "unit": "iter/sec",
            "range": "stddev: 0.000011925025986197837",
            "extra": "mean: 226.8773219936274 usec\nrounds: 2528"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_small",
            "value": 5.902294133549787,
            "unit": "iter/sec",
            "range": "stddev: 0.0015473516269241406",
            "extra": "mean: 169.42564659998993 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_large",
            "value": 1.354666468800105,
            "unit": "iter/sec",
            "range": "stddev: 0.009413759337526148",
            "extra": "mean: 738.1890842000018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_realistic",
            "value": 3.5668155909908017,
            "unit": "iter/sec",
            "range": "stddev: 0.0006867888322525345",
            "extra": "mean: 280.3621253999893 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_rolling_median",
            "value": 1310.5797819798684,
            "unit": "iter/sec",
            "range": "stddev: 0.000023124411136531684",
            "extra": "mean: 763.0210794869113 usec\nrounds: 390"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_savgol",
            "value": 2847.320115653804,
            "unit": "iter/sec",
            "range": "stddev: 0.000014870563145617242",
            "extra": "mean: 351.20743695177356 usec\nrounds: 1824"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_methods_comparison_benchmark",
            "value": 542.9286094177405,
            "unit": "iter/sec",
            "range": "stddev: 0.0001408431506813059",
            "extra": "mean: 1.8418627838979458 msec\nrounds: 472"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_planet_observation_initialization_benchmark",
            "value": 7061.853752321344,
            "unit": "iter/sec",
            "range": "stddev: 0.000012939053672183466",
            "extra": "mean: 141.6058778718384 usec\nrounds: 2481"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_detect_limb_gradient_break_benchmark",
            "value": 3.5796900896049006,
            "unit": "iter/sec",
            "range": "stddev: 0.0005214430980246897",
            "extra": "mean: 279.353791799997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_complete_geometry_pipeline_benchmark",
            "value": 3680.5669140777927,
            "unit": "iter/sec",
            "range": "stddev: 0.000010787464951006602",
            "extra": "mean: 271.6972747255598 usec\nrounds: 2639"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_image_processing_pipeline_benchmark",
            "value": 3.8857347110899574,
            "unit": "iter/sec",
            "range": "stddev: 0.0007807144461658213",
            "extra": "mean: 257.351588399996 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_parameter_optimization_simulation_benchmark",
            "value": 535.9106571955322,
            "unit": "iter/sec",
            "range": "stddev: 0.000026152507932382464",
            "extra": "mean: 1.8659826718749883 msec\nrounds: 512"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "100499183+bogsdarking@users.noreply.github.com",
            "name": "Brandon Anderson",
            "username": "bogsdarking"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "05cc800bc296ed880f1a8904dcd1f38e886e2c4f",
          "message": "v1.5.0\n\n### Added\n\n- Fit Dashboard -- an easy-to-read interface that shows status, warnings, hints and recent output\n- New tutorial notebook for measuring your own photos: see notebooks/tutorials/measure_your_planet.ipynb\n\n### Changed\n\n- Renamed gradient smoothing parameters to be more distinct\n- Reworked tutorials to move sequentially through demo, pre-configured, auto-configured, then advanced fits.",
          "timestamp": "2025-11-11T21:23:04-05:00",
          "tree_id": "ccc603b068a222eb633ab070157b434284cdbe9b",
          "url": "https://github.com/bogsdarking/planet_ruler/commit/05cc800bc296ed880f1a8904dcd1f38e886e2c4f"
        },
        "date": 1762914677169,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_benchmark",
            "value": 644565.4075010776,
            "unit": "iter/sec",
            "range": "stddev: 6.036904504182891e-7",
            "extra": "mean: 1.551432931960948 usec\nrounds: 35054"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_camera_angle_benchmark",
            "value": 1089200.9058329219,
            "unit": "iter/sec",
            "range": "stddev: 1.9862120669265285e-7",
            "extra": "mean: 918.1042676743743 nsec\nrounds: 78778"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_vectorized_benchmark",
            "value": 996.9923827738692,
            "unit": "iter/sec",
            "range": "stddev: 0.00001758633665664367",
            "extra": "mean: 1.0030166902757702 msec\nrounds: 946"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_small",
            "value": 8233.192179774467,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036545395331435734",
            "extra": "mean: 121.45957220051109 usec\nrounds: 3331"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_large",
            "value": 4211.385867351769,
            "unit": "iter/sec",
            "range": "stddev: 0.000007426111873411487",
            "extra": "mean: 237.4515258153788 usec\nrounds: 2944"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_coordinate_transforms_benchmark",
            "value": 8773.436992273198,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034871612423492093",
            "extra": "mean: 113.98041621324732 usec\nrounds: 1838"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_packing_benchmark",
            "value": 1301398.0564670851,
            "unit": "iter/sec",
            "range": "stddev: 6.615405846143234e-8",
            "extra": "mean: 768.4044055780408 nsec\nrounds: 70127"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_unpacking_benchmark",
            "value": 680738.6785282738,
            "unit": "iter/sec",
            "range": "stddev: 3.4230495009507897e-7",
            "extra": "mean: 1.4689924805829377 usec\nrounds: 115168"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_evaluation_benchmark",
            "value": 41654.09148011494,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010759843043097432",
            "extra": "mean: 24.007245494176377 usec\nrounds: 23414"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_different_losses_benchmark",
            "value": 5062.68166393116,
            "unit": "iter/sec",
            "range": "stddev: 0.000007873681378063156",
            "extra": "mean: 197.52377620826005 usec\nrounds: 2690"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_small",
            "value": 7.986745662412309,
            "unit": "iter/sec",
            "range": "stddev: 0.0012328423897303667",
            "extra": "mean: 125.20744271427833 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_large",
            "value": 1.7319506187334195,
            "unit": "iter/sec",
            "range": "stddev: 0.002337518045952025",
            "extra": "mean: 577.3836673999995 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_realistic",
            "value": 4.8043180688094465,
            "unit": "iter/sec",
            "range": "stddev: 0.00010894690833222732",
            "extra": "mean: 208.14608559999215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_rolling_median",
            "value": 1352.441091394611,
            "unit": "iter/sec",
            "range": "stddev: 0.00012297356197744672",
            "extra": "mean: 739.4037391815856 usec\nrounds: 855"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_savgol",
            "value": 3374.1482074580804,
            "unit": "iter/sec",
            "range": "stddev: 0.000020991405319652098",
            "extra": "mean: 296.3710953151496 usec\nrounds: 1857"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_methods_comparison_benchmark",
            "value": 589.9033013375536,
            "unit": "iter/sec",
            "range": "stddev: 0.000028422704345770236",
            "extra": "mean: 1.695193089668406 msec\nrounds: 513"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_planet_observation_initialization_benchmark",
            "value": 3431.8594560338847,
            "unit": "iter/sec",
            "range": "stddev: 0.00000974869667422391",
            "extra": "mean: 291.3872239848876 usec\nrounds: 1576"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_detect_limb_gradient_break_benchmark",
            "value": 4.782741064873148,
            "unit": "iter/sec",
            "range": "stddev: 0.0005371846217139466",
            "extra": "mean: 209.0851222000083 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_complete_geometry_pipeline_benchmark",
            "value": 4802.88150668562,
            "unit": "iter/sec",
            "range": "stddev: 0.000005267104733979437",
            "extra": "mean: 208.20834297244235 usec\nrounds: 2974"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_image_processing_pipeline_benchmark",
            "value": 5.074227843035937,
            "unit": "iter/sec",
            "range": "stddev: 0.0003804923466605818",
            "extra": "mean: 197.07431966667363 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_parameter_optimization_simulation_benchmark",
            "value": 550.6447540778386,
            "unit": "iter/sec",
            "range": "stddev: 0.000012708233062456516",
            "extra": "mean: 1.8160528954365396 msec\nrounds: 526"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "100499183+bogsdarking@users.noreply.github.com",
            "name": "Brandon Anderson",
            "username": "bogsdarking"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "71b769f9494559d811ca9ec12acbe8b78d0ca3df",
          "message": "v1.6.0\n\n### Added\n\n- Expanded test benchmark suite including end-to-end detection and measurement methods.\n- Added 'plot_residuals' function to zoom in on fit quality along with a 'plot_gradient_field_quiver' to directly visualize the field.\n- Added 'plot_sam_masks' and generic 'plot_segmentation_masks' to visualize segmentation output.\n- New tutorial 1.5 specifically for taking limb measurements from an airplane.\n- New tutorial 4 on selecting a detection method.\n- New manual annotation step available for ML segmentation -- user can tag masks to increase accuracy.\n\n### Changed\n\n- ImageSegmentation class replaced by the more method-agnostic MaskSegmenter",
          "timestamp": "2025-12-03T07:27:01-05:00",
          "tree_id": "17ad6e12eec05a8b603ce7c81c921cebc0e4a032",
          "url": "https://github.com/bogsdarking/planet_ruler/commit/71b769f9494559d811ca9ec12acbe8b78d0ca3df"
        },
        "date": 1764767966836,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_benchmark",
            "value": 627077.4930399301,
            "unit": "iter/sec",
            "range": "stddev: 3.534556583600612e-7",
            "extra": "mean: 1.5946992374933213 usec\nrounds: 31992"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_camera_angle_benchmark",
            "value": 1014397.1579160016,
            "unit": "iter/sec",
            "range": "stddev: 2.898915887258289e-7",
            "extra": "mean: 985.8071783781617 nsec\nrounds: 56809"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_vectorized_benchmark",
            "value": 847.9765016044822,
            "unit": "iter/sec",
            "range": "stddev: 0.00006184535262592831",
            "extra": "mean: 1.1792779612499515 msec\nrounds: 800"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_small",
            "value": 6231.7573892831615,
            "unit": "iter/sec",
            "range": "stddev: 0.000009289694584200435",
            "extra": "mean: 160.46837794419173 usec\nrounds: 1868"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_large",
            "value": 3078.788680030117,
            "unit": "iter/sec",
            "range": "stddev: 0.000038006953775448406",
            "extra": "mean: 324.8030650775999 usec\nrounds: 2643"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_coordinate_transforms_benchmark",
            "value": 8241.974324581563,
            "unit": "iter/sec",
            "range": "stddev: 0.000005867681428527098",
            "extra": "mean: 121.33015229342747 usec\nrounds: 5450"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_packing_benchmark",
            "value": 1390055.0133485661,
            "unit": "iter/sec",
            "range": "stddev: 5.361729371648353e-8",
            "extra": "mean: 719.3959882142117 nsec\nrounds: 68037"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_unpacking_benchmark",
            "value": 654733.415531883,
            "unit": "iter/sec",
            "range": "stddev: 4.046640704770384e-7",
            "extra": "mean: 1.5273391830591605 usec\nrounds: 138639"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_evaluation_benchmark",
            "value": 41990.72528422624,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016424363533513778",
            "extra": "mean: 23.81478274621869 usec\nrounds: 26232"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_different_losses_benchmark",
            "value": 4429.250772771089,
            "unit": "iter/sec",
            "range": "stddev: 0.000012068213304832055",
            "extra": "mean: 225.77181814755684 usec\nrounds: 2656"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_small",
            "value": 5.935984919923135,
            "unit": "iter/sec",
            "range": "stddev: 0.0007134734793567978",
            "extra": "mean: 168.46403983333383 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_large",
            "value": 1.376408097037411,
            "unit": "iter/sec",
            "range": "stddev: 0.004609127139018675",
            "extra": "mean: 726.5287106000073 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_realistic",
            "value": 3.6089053052826148,
            "unit": "iter/sec",
            "range": "stddev: 0.001082958604305682",
            "extra": "mean: 277.09233560000257 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_rolling_median",
            "value": 1325.1202683718188,
            "unit": "iter/sec",
            "range": "stddev: 0.00008607601725705262",
            "extra": "mean: 754.6484827590061 usec\nrounds: 783"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_savgol",
            "value": 2850.449386426469,
            "unit": "iter/sec",
            "range": "stddev: 0.00001402790871377084",
            "extra": "mean: 350.8218755828086 usec\nrounds: 1929"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_methods_comparison_benchmark",
            "value": 559.5555644016164,
            "unit": "iter/sec",
            "range": "stddev: 0.000030489473942215814",
            "extra": "mean: 1.7871326167033847 msec\nrounds: 467"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_planet_observation_initialization_benchmark",
            "value": 7306.541142838568,
            "unit": "iter/sec",
            "range": "stddev: 0.000012649378090167802",
            "extra": "mean: 136.86366509824418 usec\nrounds: 2338"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_detect_limb_gradient_break_benchmark",
            "value": 3.567541189977975,
            "unit": "iter/sec",
            "range": "stddev: 0.0019481281068019253",
            "extra": "mean: 280.30510279999703 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_complete_geometry_pipeline_benchmark",
            "value": 3663.545002741792,
            "unit": "iter/sec",
            "range": "stddev: 0.000012081144057670113",
            "extra": "mean: 272.95966045226726 usec\nrounds: 2521"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_image_processing_pipeline_benchmark",
            "value": 3.9308272208118464,
            "unit": "iter/sec",
            "range": "stddev: 0.00058084297421666",
            "extra": "mean: 254.3993779999994 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_parameter_optimization_simulation_benchmark",
            "value": 538.6917181224263,
            "unit": "iter/sec",
            "range": "stddev: 0.000027920071351484383",
            "extra": "mean: 1.8563493114121608 msec\nrounds: 517"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_extract_exif_benchmark",
            "value": 1022.4002930366867,
            "unit": "iter/sec",
            "range": "stddev: 0.0000369235490952471",
            "extra": "mean: 978.0904864863111 usec\nrounds: 185"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_extract_camera_parameters_benchmark",
            "value": 897.252334095657,
            "unit": "iter/sec",
            "range": "stddev: 0.000060786246993482826",
            "extra": "mean: 1.114513679151253 msec\nrounds: 801"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_camera_model_detection_benchmark",
            "value": 1063.8858032878545,
            "unit": "iter/sec",
            "range": "stddev: 0.00008627367932578695",
            "extra": "mean: 939.9505068209195 usec\nrounds: 953"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_focal_length_extraction_benchmark",
            "value": 1077.1616786751927,
            "unit": "iter/sec",
            "range": "stddev: 0.000013714706179135331",
            "extra": "mean: 928.365740999908 usec\nrounds: 1000"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_gps_altitude_extraction_benchmark",
            "value": 1080.042716738985,
            "unit": "iter/sec",
            "range": "stddev: 0.00001731374923076976",
            "extra": "mean: 925.8893046557815 usec\nrounds: 988"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestConfigurationBenchmarks::test_create_config_from_image_benchmark",
            "value": 480.7621712407576,
            "unit": "iter/sec",
            "range": "stddev: 0.00002328401037881505",
            "extra": "mean: 2.0800305427924712 msec\nrounds: 444"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestConfigurationBenchmarks::test_config_validation_benchmark",
            "value": 446978.81360148086,
            "unit": "iter/sec",
            "range": "stddev: 4.6645982009624533e-7",
            "extra": "mean: 2.2372425036046204 usec\nrounds: 64898"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestLimbDetectionBenchmarks::test_gradient_field_detection_benchmark",
            "value": 0.015390683750981649,
            "unit": "iter/sec",
            "range": "stddev: 1.1600620332704938",
            "extra": "mean: 64.9743712612 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestLimbDetectionBenchmarks::test_manual_limb_detection_benchmark",
            "value": 0.06946570656965857,
            "unit": "iter/sec",
            "range": "stddev: 0.04671842436150711",
            "extra": "mean: 14.395592435200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestUncertaintyBenchmarks::test_parameter_uncertainty_calculation_benchmark",
            "value": 5309.318581949208,
            "unit": "iter/sec",
            "range": "stddev: 0.000011638699210885074",
            "extra": "mean: 188.34808734209926 usec\nrounds: 2702"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestUncertaintyBenchmarks::test_multiple_parameter_uncertainties_benchmark",
            "value": 1782.323964963515,
            "unit": "iter/sec",
            "range": "stddev: 0.000017379635844286857",
            "extra": "mean: 561.0652270057259 usec\nrounds: 1370"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFullPipelineBenchmarks::test_full_pipeline_earth_benchmark",
            "value": 0.009642026417455872,
            "unit": "iter/sec",
            "range": "stddev: 0.264389188833206",
            "extra": "mean: 103.71263847499998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFullPipelineBenchmarks::test_configuration_generation_workflow_benchmark",
            "value": 308.1052275158446,
            "unit": "iter/sec",
            "range": "stddev: 0.00003747137528949836",
            "extra": "mean: 3.245644379560467 msec\nrounds: 274"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFullPipelineBenchmarks::test_observation_creation_and_setup_benchmark",
            "value": 36.23381322820098,
            "unit": "iter/sec",
            "range": "stddev: 0.0003142574828636492",
            "extra": "mean: 27.598530513528573 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_custom_backend_segment_benchmark",
            "value": 79.85078884710467,
            "unit": "iter/sec",
            "range": "stddev: 0.000861190004699519",
            "extra": "mean: 12.523357808208546 msec\nrounds: 73"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_classify_automatic_benchmark",
            "value": 2106.0675852257204,
            "unit": "iter/sec",
            "range": "stddev: 0.000010007577879315873",
            "extra": "mean: 474.8185704082349 usec\nrounds: 1960"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_combine_masks_benchmark",
            "value": 47.52198754653737,
            "unit": "iter/sec",
            "range": "stddev: 0.0000953785640854207",
            "extra": "mean: 21.04289091487849 msec\nrounds: 47"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_downsampling_benchmark_2x",
            "value": 108.81799433054786,
            "unit": "iter/sec",
            "range": "stddev: 0.00005097724214073024",
            "extra": "mean: 9.189656601851883 msec\nrounds: 108"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_downsampling_benchmark_4x",
            "value": 33.86057599700224,
            "unit": "iter/sec",
            "range": "stddev: 0.00019408940520125828",
            "extra": "mean: 29.53287032354477 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_outlier_detection_benchmark",
            "value": 187.7405673380816,
            "unit": "iter/sec",
            "range": "stddev: 0.0001792089835869757",
            "extra": "mean: 5.3264992972947 msec\nrounds: 185"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_sparse_mask_interpolation_benchmark",
            "value": 290.68410296789864,
            "unit": "iter/sec",
            "range": "stddev: 0.000038099023424362096",
            "extra": "mean: 3.440160606617122 msec\nrounds: 272"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_full_pipeline_benchmark_large_image",
            "value": 3.5918671240584263,
            "unit": "iter/sec",
            "range": "stddev: 0.004425916874618214",
            "extra": "mean: 278.4067354000854 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationComparisonBenchmarks::test_downsampling_speedup_comparison",
            "value": 35.605135240435075,
            "unit": "iter/sec",
            "range": "stddev: 0.0002200159638700065",
            "extra": "mean: 28.085836305554796 msec\nrounds: 36"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "100499183+bogsdarking@users.noreply.github.com",
            "name": "Brandon Anderson",
            "username": "bogsdarking"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9d9500a2d928f58ac7f1e4ff413a341977df9a5",
          "message": "v1.6.1\n\n### Fixed\n\n- Added missing requirement to build mermaid diagrams -- see new tutorials!",
          "timestamp": "2025-12-04T22:22:05-05:00",
          "tree_id": "64abd0b0581374f2502f2d1879283c87cde175f3",
          "url": "https://github.com/bogsdarking/planet_ruler/commit/f9d9500a2d928f58ac7f1e4ff413a341977df9a5"
        },
        "date": 1764908139065,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_benchmark",
            "value": 603314.8424182354,
            "unit": "iter/sec",
            "range": "stddev: 3.9981869584398074e-7",
            "extra": "mean: 1.6575093627595043 usec\nrounds: 30065"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_camera_angle_benchmark",
            "value": 984976.8226197428,
            "unit": "iter/sec",
            "range": "stddev: 3.051808885886607e-7",
            "extra": "mean: 1.015252315623326 usec\nrounds: 54630"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_horizon_distance_vectorized_benchmark",
            "value": 846.1251684633617,
            "unit": "iter/sec",
            "range": "stddev: 0.0000214400440167403",
            "extra": "mean: 1.1818582371401252 msec\nrounds: 797"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_small",
            "value": 6197.418820078671,
            "unit": "iter/sec",
            "range": "stddev: 0.000008775298440194571",
            "extra": "mean: 161.35749882840835 usec\nrounds: 1706"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_limb_arc_benchmark_large",
            "value": 3079.2124813084956,
            "unit": "iter/sec",
            "range": "stddev: 0.000009321681422312502",
            "extra": "mean: 324.7583614545025 usec\nrounds: 2584"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestGeometryBenchmarks::test_coordinate_transforms_benchmark",
            "value": 8176.874738061072,
            "unit": "iter/sec",
            "range": "stddev: 0.000006971937108818587",
            "extra": "mean: 122.29611337266532 usec\nrounds: 4913"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_packing_benchmark",
            "value": 1286280.9450302757,
            "unit": "iter/sec",
            "range": "stddev: 9.24830579386387e-8",
            "extra": "mean: 777.4351348852864 nsec\nrounds: 185529"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_parameter_unpacking_benchmark",
            "value": 644139.5184291835,
            "unit": "iter/sec",
            "range": "stddev: 4.165746044027972e-7",
            "extra": "mean: 1.5524587009327229 usec\nrounds: 122775"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_evaluation_benchmark",
            "value": 41292.63549110562,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030527925790899097",
            "extra": "mean: 24.217393443327172 usec\nrounds: 24555"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFitBenchmarks::test_cost_function_different_losses_benchmark",
            "value": 4389.110852821847,
            "unit": "iter/sec",
            "range": "stddev: 0.00002408378212833286",
            "extra": "mean: 227.8365786448706 usec\nrounds: 2613"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_small",
            "value": 5.911223258207606,
            "unit": "iter/sec",
            "range": "stddev: 0.0012418236359253018",
            "extra": "mean: 169.169722800018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_large",
            "value": 1.3593541314417676,
            "unit": "iter/sec",
            "range": "stddev: 0.007983312571822833",
            "extra": "mean: 735.6434771999943 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_gradient_break_benchmark_realistic",
            "value": 3.5892920910350394,
            "unit": "iter/sec",
            "range": "stddev: 0.0006410292129741081",
            "extra": "mean: 278.60647019998623 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_rolling_median",
            "value": 1341.9212280414029,
            "unit": "iter/sec",
            "range": "stddev: 0.000019325518611389346",
            "extra": "mean: 745.2002242035824 usec\nrounds: 785"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_benchmark_savgol",
            "value": 2832.8721946992127,
            "unit": "iter/sec",
            "range": "stddev: 0.000022066858967990183",
            "extra": "mean: 352.99862869605295 usec\nrounds: 1826"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestImageProcessingBenchmarks::test_smooth_limb_methods_comparison_benchmark",
            "value": 557.7712352925589,
            "unit": "iter/sec",
            "range": "stddev: 0.000030470207334276803",
            "extra": "mean: 1.7928497145886089 msec\nrounds: 473"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_planet_observation_initialization_benchmark",
            "value": 7302.386023225281,
            "unit": "iter/sec",
            "range": "stddev: 0.000013338059442884109",
            "extra": "mean: 136.9415416850731 usec\nrounds: 2291"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestObservationBenchmarks::test_detect_limb_gradient_break_benchmark",
            "value": 3.591356343376891,
            "unit": "iter/sec",
            "range": "stddev: 0.004413721593936775",
            "extra": "mean: 278.4463318000121 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_complete_geometry_pipeline_benchmark",
            "value": 3654.5288331572524,
            "unit": "iter/sec",
            "range": "stddev: 0.000013408780336919277",
            "extra": "mean: 273.63308531788795 usec\nrounds: 2520"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_image_processing_pipeline_benchmark",
            "value": 3.9042635716988014,
            "unit": "iter/sec",
            "range": "stddev: 0.0007553229040448631",
            "extra": "mean: 256.1302488000024 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestIntegratedWorkflowBenchmarks::test_parameter_optimization_simulation_benchmark",
            "value": 528.7719695072705,
            "unit": "iter/sec",
            "range": "stddev: 0.00003547592808194989",
            "extra": "mean: 1.8911743769849174 msec\nrounds: 504"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_extract_exif_benchmark",
            "value": 1009.2401363480758,
            "unit": "iter/sec",
            "range": "stddev: 0.00006274056261801607",
            "extra": "mean: 990.8444620707307 usec\nrounds: 145"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_extract_camera_parameters_benchmark",
            "value": 885.5056421251467,
            "unit": "iter/sec",
            "range": "stddev: 0.00009117595296313964",
            "extra": "mean: 1.1292982816010924 msec\nrounds: 799"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_camera_model_detection_benchmark",
            "value": 1050.4753367060273,
            "unit": "iter/sec",
            "range": "stddev: 0.00003088036637454891",
            "extra": "mean: 951.9500030678468 usec\nrounds: 978"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_focal_length_extraction_benchmark",
            "value": 1056.6811859567872,
            "unit": "iter/sec",
            "range": "stddev: 0.00001585831998506431",
            "extra": "mean: 946.3592361536518 usec\nrounds: 957"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestCameraParameterBenchmarks::test_gps_altitude_extraction_benchmark",
            "value": 1068.6081860748461,
            "unit": "iter/sec",
            "range": "stddev: 0.000017776225235886596",
            "extra": "mean: 935.7966867848412 usec\nrounds: 961"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestConfigurationBenchmarks::test_create_config_from_image_benchmark",
            "value": 475.76160112551423,
            "unit": "iter/sec",
            "range": "stddev: 0.00008939760788982712",
            "extra": "mean: 2.101893043983141 msec\nrounds: 432"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestConfigurationBenchmarks::test_config_validation_benchmark",
            "value": 453674.5388953536,
            "unit": "iter/sec",
            "range": "stddev: 6.048029261605651e-7",
            "extra": "mean: 2.204223323695633 usec\nrounds: 63943"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestLimbDetectionBenchmarks::test_gradient_field_detection_benchmark",
            "value": 0.015017047778251046,
            "unit": "iter/sec",
            "range": "stddev: 1.117240973010312",
            "extra": "mean: 66.59098477719996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestLimbDetectionBenchmarks::test_manual_limb_detection_benchmark",
            "value": 0.06909460354418517,
            "unit": "iter/sec",
            "range": "stddev: 0.036510196531317124",
            "extra": "mean: 14.472910309999998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestUncertaintyBenchmarks::test_parameter_uncertainty_calculation_benchmark",
            "value": 5243.968697154646,
            "unit": "iter/sec",
            "range": "stddev: 0.000013759606832903734",
            "extra": "mean: 190.69526493218683 usec\nrounds: 2461"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestUncertaintyBenchmarks::test_multiple_parameter_uncertainties_benchmark",
            "value": 1759.5025750901445,
            "unit": "iter/sec",
            "range": "stddev: 0.00001867249272178415",
            "extra": "mean: 568.342447551557 usec\nrounds: 1287"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFullPipelineBenchmarks::test_full_pipeline_earth_benchmark",
            "value": 0.009632668585335402,
            "unit": "iter/sec",
            "range": "stddev: 0.3855594401990141",
            "extra": "mean: 103.81339201500005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFullPipelineBenchmarks::test_configuration_generation_workflow_benchmark",
            "value": 309.4061125581627,
            "unit": "iter/sec",
            "range": "stddev: 0.00004111680663395171",
            "extra": "mean: 3.231998203694241 msec\nrounds: 270"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestFullPipelineBenchmarks::test_observation_creation_and_setup_benchmark",
            "value": 35.496812068336965,
            "unit": "iter/sec",
            "range": "stddev: 0.00024799782719505477",
            "extra": "mean: 28.17154391427721 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_custom_backend_segment_benchmark",
            "value": 81.97121626646677,
            "unit": "iter/sec",
            "range": "stddev: 0.0007831598919108371",
            "extra": "mean: 12.199404199997035 msec\nrounds: 90"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_classify_automatic_benchmark",
            "value": 2097.404473021674,
            "unit": "iter/sec",
            "range": "stddev: 0.000019024065836499602",
            "extra": "mean: 476.7797593943942 usec\nrounds: 1916"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_combine_masks_benchmark",
            "value": 46.54694729757951,
            "unit": "iter/sec",
            "range": "stddev: 0.0002882567889867203",
            "extra": "mean: 21.48368600000544 msec\nrounds: 45"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_downsampling_benchmark_2x",
            "value": 106.55603909393471,
            "unit": "iter/sec",
            "range": "stddev: 0.0001385078360305423",
            "extra": "mean: 9.384733221159317 msec\nrounds: 104"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_downsampling_benchmark_4x",
            "value": 33.545755383235594,
            "unit": "iter/sec",
            "range": "stddev: 0.0003253666275430703",
            "extra": "mean: 29.810030764719265 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_outlier_detection_benchmark",
            "value": 188.92479654037177,
            "unit": "iter/sec",
            "range": "stddev: 0.000038976766163886174",
            "extra": "mean: 5.293111430115038 msec\nrounds: 186"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_sparse_mask_interpolation_benchmark",
            "value": 290.22085923376153,
            "unit": "iter/sec",
            "range": "stddev: 0.00003513964644821132",
            "extra": "mean: 3.445651710356695 msec\nrounds: 290"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationBenchmarks::test_full_pipeline_benchmark_large_image",
            "value": 4.781016474566868,
            "unit": "iter/sec",
            "range": "stddev: 0.0006548635655308757",
            "extra": "mean: 209.16054260001147 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark_performance.py::TestSegmentationComparisonBenchmarks::test_downsampling_speedup_comparison",
            "value": 34.93577771026657,
            "unit": "iter/sec",
            "range": "stddev: 0.00035069295591370485",
            "extra": "mean: 28.623951305544576 msec\nrounds: 36"
          }
        ]
      }
    ]
  }
}