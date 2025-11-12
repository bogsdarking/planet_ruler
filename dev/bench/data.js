window.BENCHMARK_DATA = {
  "lastUpdate": 1762914677977,
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
      }
    ]
  }
}