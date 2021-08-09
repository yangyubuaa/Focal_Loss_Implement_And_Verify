### 验证focal loss的性能
1. 验证focal loss在均衡数据集下与cross entropy的对比
2. 验证focal loss在样本不均衡下与cross entropy的对比（不均衡但差不太多）
3. 验证focal loss在样本极度不均衡下与cross entropy的对比（10:1）
---
先说结论：
focal loss只能一定程度上提高模型的效果，在样本不均衡时可以使用focal loss，样本均衡时使用cross entropy loss即可，尤其是数据存在大量噪声的情况，避免focal loss关注到噪声数据。

分析：
1. 样本均衡情况下，不用focal loss，指标稳步增长，使用focal loss，低的指标会迅速上来尽量做平衡
2. 样本不均衡情况下，不用focal loss，指标稳步增长，数据量大的类别指标初值会比较高（精度高，召回低），数据量小的类别（精度低，召回率高）。但是指标都在稳步增长，直到逼近上限。使用focal loss的话，数据量小的类别精度会迅速提升。最终逼近上限。

focal loss和cross entropy loss最终都会逼近模型的上限，为什么focal loss能够起到一定的效果？因为同样是逼近到了一个上限，但是过程是不同的，cross entropy loss在逼近上限的过程中重复学习了大量样本类别的数据，可能会导致在大样本上过拟合，但是focal loss不会，使用focal loss逼近同样效果，各个类别对于模型的权重更新贡献是一样的，不会出现多样本贡献的权重大的情况，因为模型关注难分样本，对易分样本减小权重，这样就会减轻数据不均衡带来的影响。
