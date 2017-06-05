Human Activity Analysis:
=====


Create QSRs from Leeds Human Activity Dataset.
---


- Download dataset from [here](http://doi.org/10.5518/86) and put it in: `~/Datasets/ECAI_Data/`


- Use the segmentation code to create separate video clips each with a GT lable.

  ```
  python prepare_folders.py
  ```
  note: move script to `~/Datasets/ECAI_Data/` also.


- Then create QSRs:

  ```
  rosrun ch6_2 qsr_ecai.py
  ```

- Then run LDA/Online LDA using:

  ```
  rosrun ch6_2 learn_topics.py <RUN_NUMBER>
  ```
  Output in `~/Datasets/ECAI_Data/segmented/QSR_path/<RUN_NUMBER>`
