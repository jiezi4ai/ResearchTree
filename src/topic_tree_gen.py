## Embedding

### generate embeddings for paper titles and abstracts
from models.embedding_models import gemini_embedding_async

gemini_api_key = os.getenv('GEMINI_API_KEY_1')
gemini_embed_mdl_name = "models/text-embedding-004"

target_info = f"TITLE: {s2_metadata[0].get('title')} \nABSTRACT: {s2_metadata[0].get('abstract')}"
print(target_info)


texts = []
candit_paper_info = []
for item in extended_paper_info:
    if item.get('title') is not None and item.get('abstract') is not None:
        texts.append(f"TITLE: {item.get('title')} \nABSTRACT: {item.get('abstract')}")
        candit_paper_info.append(item)

all_embeds = await gemini_embedding_async(gemini_api_key, gemini_embed_mdl_name, texts, 10)

import numpy as np
from models.embedding_models import semantic_similarity_matrix

similarity_matrix = semantic_similarity_matrix(all_embeds[0], all_embeds[1:])
similarity_matrix = np.array(similarity_matrix)

sims = similarity_matrix[0].tolist()


### Study distribution of similarity
import matplotlib.pyplot as plt
import numpy as np

# 使用 matplotlib 创建直方图
plt.hist(sims, bins=10)  # bins 参数可以调整柱子的数量
plt.xlabel('数据值')
plt.ylabel('频数')
plt.title('数据分布直方图')
plt.grid(True) # 添加网格线，可选
plt.show()


### filter papers
filtered_paper_metadata = []
filtered_embedings = []

for idx, item in enumerate(candit_paper_info[1:]):
    if sims[idx] > 0.7:
        filtered_paper_metadata.append(item)
        filtered_embedings.append(all_embeds[idx])

print(len(filtered_paper_metadata), len(filtered_embedings))


i = 0 
for item in filtered_paper_metadata:
    if item.get('relationship') in ['related', 'recommended']:
        print(item.get('query'), item.get('title'))
        i += 1
print(i)


## Clustering Papers
### Basic Clustering
from sklearn.cluster import KMeans

num_clusters = 5  #  设置聚类的数量， 这通常需要根据实际情况调整
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10) # 显式设置 n_init
clusters = kmeans.fit_predict(filtered_embedings)

print("聚类结果:")
for i in range(num_clusters):
    print("="*40)
    print(f"\nCluster {i}:")
    for sentence_id, cluster_id in enumerate(clusters):
        if cluster_id == i:
            print(f"- {filtered_paper_metadata[sentence_id].get('title')}")


### Automatic clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

# 步骤 4: 尝试不同的 k 值并评估聚类指标
silhouette_scores = []
davies_bouldin_scores = []
inertia_values = []  # 用于 Elbow Method
k_values = range(2, 7)  #  尝试 k 从 2 到 6， 你可以根据需要调整范围

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(filtered_embedings)
    inertia_values.append(kmeans.inertia_) #  inertia_ 对应 KMeans 的 Within-Cluster Sum of Squares (WCSS)

    #  计算 Silhouette Score,  注意需要处理簇的数量少于 2 的情况
    if len(set(clusters)) > 1: # 确保簇的数量大于 1 才能计算 Silhouette Score
        silhouette_avg = silhouette_score(filtered_embedings, clusters)
        silhouette_scores.append(silhouette_avg)
    else:
        silhouette_scores.append(np.nan) #  如果簇的数量为 1,  Silhouette Score  无法计算， 用 NaN 填充

    # 计算 Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(filtered_embedings, clusters)
    davies_bouldin_scores.append(davies_bouldin)


# 步骤 5:  可视化评估指标和 Elbow Method

plt.figure(figsize=(15, 5))

#  子图 1: Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values) # 确保 x 轴刻度是整数 k 值
plt.grid(True)

# 子图 2: Davies-Bouldin Index
plt.subplot(1, 3, 2)
plt.plot(k_values, davies_bouldin_scores, marker='o')
plt.title('Davies-Bouldin Index for Different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.xticks(k_values)
plt.grid(True)


# 子图 3: Elbow Method (Inertia)
plt.subplot(1, 3, 3)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.xticks(k_values)
plt.grid(True)


plt.tight_layout()  # 调整子图布局，避免重叠
plt.show()


# 步骤 6:  基于指标选择最优的 k 值 (简单的选择方法， 你可以根据需要调整)
#  这里我们简单地选择 Silhouette Score 最高， Davies-Bouldin Index 最低的 k 值作为参考
optimal_k_silhouette = k_values[np.nanargmax(silhouette_scores)] # nanargmax 处理 NaN 值
optimal_k_db = k_values[np.argmin(davies_bouldin_scores)]


print("\n聚类评估指标:")
for i, k in enumerate(k_values):
    sil_score = silhouette_scores[i] if not np.isnan(silhouette_scores[i]) else "N/A" #  处理 NaN 值
    print(f"k={k}: Silhouette Score: {sil_score:.4f}, Davies-Bouldin Index: {davies_bouldin_scores[i]:.4f}, Inertia: {inertia_values[i]:.4f}")
    
print(f"\n基于 Silhouette Score, 建议的最优 k 值为: {optimal_k_silhouette}")
print(f"基于 Davies-Bouldin Index, 建议的最优 k 值为: {optimal_k_db}")
print("\n请参考 Elbow Method 图表， 综合评估并选择最终的 k 值。")


#  步骤 7:  使用最优的 k 值重新聚类并展示结果 (可选， 这里使用 Silhouette Score 建议的 k 值)
best_k = optimal_k_silhouette #  你可以根据你的评估选择 optimal_k_db 或手动选择
kmeans_optimal = KMeans(n_clusters=best_k, random_state=0, n_init=10)
optimal_clusters = kmeans_optimal.fit_predict(filtered_embedings)

print(f"\n使用最优 k 值 ({best_k}) 的聚类结果:")
for i in range(best_k):
    print(f"\nCluster {i}:")
    for sentence_id, cluster_id in enumerate(optimal_clusters):
        if cluster_id == i:
            print(f"- {filtered_paper_metadata[sentence_id].get('title')}")