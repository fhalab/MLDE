Pairwise XGBoost Comparisons
=============================
This file displays additional data corresponding to the section "Models/Training Procedures More Tailored for Combinatorial Fitness Landscapes Improve MLDE Predictive Performance" of the manuscript associated with this  repository. When comparing the different learning objectives (Tweedie vs root mean squared) of XGBoost, we used the same training variants, cross-validation indices, and random seeds for a given simulation. This allows us to make pairwise comparisons between the different summary metrics. Figures giving these pairwise comparisons are displayed in this file. Results for all encodings tested in our work using either a Tree or Linear base XGBoost model are provided. Each figure gives the results of simulations at a given training size and using a given base model. Each figure contains a grid of 2x5 subplots. Each subplot gives the results of an encoding strategy using both the Tweedie learning objective (y-axis) and standard learning objective (x-axis). The diagonal line gives x = y. The numbers on either side of the line give the respective number of points falling on that side; if the total of the two numbers doesn't equal "2000", that is because a certain number of simulations yielded equivalent results.

Table of Contents
-----------------
- [Tree Base Model Results](#tree-base-model-results)
    - [Pairwise Comparisons of Max Fitness Achieved: Tree Base Model](#pairwise-comparisons-of-max-fitness-achieved-tree-base-model)
    - [Pairwise Comparisons of Mean Fitness Achieved: Tree Base Model](#pairwise-comparisons-of-mean-fitness-achieved-tree-base-model)
    - [Pairwise Comparisons of NDCG: Tree Base Model](#pairwise-comparisons-of-ndcg-tree-base-model)
- [Linear Base Model Results](#linear-base-model-results)
    - [Pairwise Comparisons of Max Fitness Achieved: Linear Base Model](#pairwise-comparisons-of-max-fitness-achieved-linear-base-model)
    - [Pairwise Comparisons of Mean Fitness Achieved: Linear Base Model](#pairwise-comparisons-of-mean-fitness-achieved-linear-base-model)
    - [Pairwise Comparisons of NDCG: Linear Base Model](#pairwise-comparisons-of-ndcg-linear-base-model)


# Tree Base Model Results
## Pairwise Comparisons of Max Fitness Achieved: Tree Base Model
![Max Fitness Achieved: Training on 384 Training Points](./images/Max_384_Tree.png)
![Max Fitness Achieved: Training on 48 Training Points](./images/Max_48_Tree.png)
![Max Fitness Achieved: Training on 24 Training Points](./images/Max_24_Tree.png)

## Pairwise Comparisons of Mean Fitness Achieved: Tree Base Model
![Mean Fitness Achieved: Training on 384 Training Points](./images/Mean_384_Tree.png)
![Mean Fitness Achieved: Training on 48 Training Points](./images/Mean_48_Tree.png)
![Mean Fitness Achieved: Training on 24 Training Points](./images/Mean_24_Tree.png)

## Pairwise Comparisons of NDCG: Tree Base Model
![NDCG: Training on 384 Training Points](./images/NDCG_384_Tree.png)
![NDCG: Training on 48 Training Points](./images/NDCG_48_Tree.png)
![NDCG: Training on 24 Training Points](./images/NDCG_24_Tree.png)

# Linear Base Model Results

## Pairwise Comparisons of Max Fitness Achieved: Linear Base Model
![Max Fitness Achieved: Training on 384 Training Points](./images/Max_384_Linear.png)
![Max Fitness Achieved: Training on 48 Training Points](./images/Max_48_Linear.png)
![Max Fitness Achieved: Training on 24 Training Points](./images/Max_24_Linear.png)

## Pairwise Comparisons of Mean Fitness Achieved: Linear Base Model
![Mean Fitness Achieved: Training on 384 Training Points](./images/Mean_384_Linear.png)
![Mean Fitness Achieved: Training on 48 Training Points](./images/Mean_48_Linear.png)
![Mean Fitness Achieved: Training on 24 Training Points](./images/Mean_24_Linear.png)

## Pairwise Comparisons of NDCG: Linear Base Model
![NDCG: Training on 384 Training Points](./images/NDCG_384_Linear.png)
![NDCG: Training on 48 Training Points](./images/NDCG_48_Linear.png)
![NDCG: Training on 24 Training Points](./images/NDCG_24_Linear.png)