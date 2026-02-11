| setting                                          |   pixel_auroc |   pixel_ap |   pixel_aupro |   image_auroc |   image_ap |   image_f1 |
|--------------------------------------------------|---------------|------------|---------------|---------------|------------|------------|
| MVTec (Best, fixed prompts + DINOv2)             |        0.9109 |     0.4084 |        0.8773 |        0.8650 |     0.9422 |     0.9043 |
| MVTec (Bayes refined, MC=8, sigma=0.001)         |        0.9110 |     0.4086 |        0.8777 |        0.8645 |     0.9418 |     0.9045 |
| VisA (Best, fixed prompts + DINOv2, map_max)     |        0.8468 |     0.1779 |        0.8089 |        0.8106 |     0.8438 |     0.8058 |
| VisA (Bayes refined, MC=8, sigma=0.001, map_max) |        0.8470 |     0.1781 |        0.8094 |        0.8112 |     0.8444 |     0.8059 |

### Cross-domain view (training-free)

The current *best* setting is **zero-shot / training-free**. Therefore, the cross-domain directions reduce to evaluating on the target domain:

| direction     |   pixel_auroc |   image_auroc |
|---------------|---------------|---------------|
| MVTec -> VisA |        0.8468 |        0.8106 |
| VisA -> MVTec |        0.9109 |        0.8650 |

