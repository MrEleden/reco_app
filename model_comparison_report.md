# Movie Recommendation Model Comparison Report

Generated on: 2025-09-28 22:15:11

## Best Models by RMSE

| Rank | Model Type | Run ID | RMSE | MAE | Accuracy | F1-Score | Status |
|------|------------|---------|------|-----|----------|----------|--------|
| 2 | hybrid | 2431f70a | 0.3255 | 0.1830 | 0.8515 | 0.5623 | FINISHED |
| 1 | hybrid | 85d8326d | 0.3288 | 0.2145 | 0.8464 | 0.4802 | FINISHED |
| 3 | hybrid | b6804d23 | 0.3304 | 0.2154 | 0.8456 | 0.5019 | FINISHED |
| 5 | collaborative | 8ab026b9 | 0.3474 | 0.2590 | 0.8339 | 0.1683 | FINISHED |
| 4 | collaborative | 252a37f2 | 0.3476 | 0.2584 | 0.8343 | 0.1675 | FINISHED |
| 6 | collaborative | f493cf1c | 0.3499 | 0.2604 | 0.8321 | 0.1427 | FINISHED |

## Summary Statistics

- **Total Models Trained**: 6
- **Best Model**: hybrid (RMSE: 0.3255)
- **Average RMSE**: 0.3383
- **Average Accuracy**: 0.8406

## Model Type Performance

              val_rmse                       val_accuracy                  val_f1                
                  mean     min     max count         mean     min     max    mean     min     max
model_type                                                                                       
collaborative   0.3483  0.3474  0.3499     3       0.8334  0.8321  0.8343  0.1595  0.1427  0.1683
hybrid          0.3282  0.3255  0.3304     3       0.8478  0.8456  0.8515  0.5148  0.4802  0.5623
