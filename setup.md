# Data
[AI vs Human on Kaggle](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/overview)


## Kaggle API
If you want to use kaggle API, you will need kagglehub. See [kaggle documentation on authentication and API](https://www.kaggle.com/docs/api#authentication).

## Data folder
Set up a folder called 'data' in your base directory. The contents of 'data' are downloaded from the above and are:
- test_data_v2
- train_data
- test.csv
- train.csv

# Environment
This environment uses the same ones as A4, but with the following additions:
- kagglehub: importing data via API. You don't need this if you are downloading directly
- albumentations: image augmentation. much faster than torchvision

---
