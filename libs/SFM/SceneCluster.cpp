/*
 * SceneCluster.cpp
 *
 * Copyright (c) 2014-2025 SEACAVE
 */

#include "Common.h"
#include "SceneCluster.h"
#include "Scene.h"
#include "Image.h"
#include "ImagePair.h"
#include "../Math/GeodeticTransforms.h"

using namespace SFM;


// S T R U C T S ///////////////////////////////////////////////////

constexpr float WEIGHT_MULTIPLIER = 10.f; // Multiplier to convert float weights to integers for METIS

SceneCluster::SceneCluster(Scene& scene, const ClusterConfig& config)
	: scene(scene), config(config)
{
}

Scene SceneCluster::ExtractSubScene(
	const IIndexArr& viewIndices,
	const IIndexArr& globalToLocal,
	unsigned nThreadsPerCluster)
{
	Scene subScene(nThreadsPerCluster);

	// Map: global camera index -> local camera index
	std::unordered_map<IIndex, IIndex> cameraMap;

	// Copy images and cameras; move expensive data (keypoints, descriptors)
	// from the global scene to sub-scenes to save memory during reconstruction.
	// These are moved back during GlobalAlignment::MergeSingleScene.
	for (IIndex globalID : viewIndices) {
		Image& img = scene.images[globalID];
		// Ensure camera exists in sub-scene
		const IIndex globalCamID = img.cameraID;
		auto ret = cameraMap.emplace(globalCamID, subScene.cameras.size());
		if (ret.second) {
			// Clone camera for independent bundle adjustment
			subScene.cameras.emplace_back(scene.cameras[globalCamID]->Clone());
		}
		IIndex localCamID = ret.first->second;
		// Add image with remapped IDs
		Image subImg = img;
		subImg.ID = subScene.images.size();
		subImg.cameraID = localCamID;
		subImg.pCamera = subScene.cameras[localCamID];
		// Move expensive data from global to sub-scene to save memory
		subImg.keypoints = std::move(img.keypoints);
		subImg.descriptors = std::move(img.descriptors);
		subScene.images.emplace_back(std::move(subImg));
	}

	// Move image pairs (only with image IDs within cluster)
	for (uint32_t i = 0; i < scene.pairs.size(); ++i) {
		ImagePair& pair = scene.pairs[i];
		const IIndex localID1 = globalToLocal[pair.ID1];
		const IIndex localID2 = globalToLocal[pair.ID2];
		if (localID1 == NO_ID || localID2 == NO_ID)
			continue; // pair crosses cluster boundary
		// Move pair to avoid copying large match vectors
		ASSERT(localID1 < localID2);
		pair.ID1 = localID1;
		pair.ID2 = localID2;
		subScene.pairs.emplace_back(std::move(pair));
		scene.pairs.RemoveAtMove(i--);
	}

	// Copy tracks (include tracks with ≥2 observations in this cluster)
	for (const Track& srcTrack : scene.tracks) {
		Track dstTrack;
		dstTrack.position = srcTrack.position;
		FOREACH(k, srcTrack.observations) {
			const Observation& obs = srcTrack.observations[k];
			const uint32_t localID = globalToLocal[obs.imageID];
			if (localID == NO_ID)
				continue; // observation outside this cluster
			dstTrack.observations.emplace_back(localID, obs.featureID);
			if (k < srcTrack.numInliers)
				++dstTrack.numInliers;
		}
		// Include track if it has at least 2 observations in this cluster
		if (dstTrack.GetNumObservations() >= 2)
			subScene.tracks.emplace_back(dstTrack);
	}

    VERBOSE("Sub-scene: %u images, %u pairs, %u tracks",
	    subScene.images.size(), subScene.pairs.size(),
	    subScene.tracks.size());
	return subScene;
}

std::vector<Scene> SceneCluster::SplitScene(std::vector<IIndexArr>* outLocalToGlobal)
{
	IIndex nViews = scene.images.size();
	if (nViews == 0) {
		return {};
	}
	if (config.maxViewsPerCluster == 0 || nViews <= config.maxViewsPerCluster) {
		// No need to split - create single cluster with identity mapping
		DEBUG("Scene has %u images, no clustering needed", nViews);
		return {std::move(scene)};
	}

	BuildConnectivityGraph();

	return SplitSceneAggregativeClustering(outLocalToGlobal);
}

void SceneCluster::BuildConnectivityGraph()
{
	const IIndex nViews = scene.images.size();
	xadj.assign(nViews + 1, 0);

	std::vector<int> degrees(nViews, 0);
	for (const ImagePair& pair : scene.pairs) {
		if (pair.GetCompositeWeight() < config.minPairWeight)
			continue;
		degrees[pair.ID1]++;
		degrees[pair.ID2]++;
	}
	for (IIndex i = 0; i < nViews; ++i) {
		xadj[i+1] = xadj[i] + degrees[i];
	}

	adjncy.assign(xadj.back(), 0);
	adjwgt.assign(xadj.back(), 0);
	std::vector<int> offsets = xadj;

	for (const ImagePair& pair : scene.pairs) {
		const float weight = pair.GetCompositeWeight();
		if (weight < config.minPairWeight)
			continue;
		const int w = cvRound(weight * WEIGHT_MULTIPLIER);

		int idx1 = offsets[pair.ID1]++;
		adjncy[idx1] = pair.ID2;
		adjwgt[idx1] = w;

		int idx2 = offsets[pair.ID2]++;
		adjncy[idx2] = pair.ID1;
		adjwgt[idx2] = w;
	}
	VERBOSE("Built connectivity graph: %u nodes, %u edges", (unsigned)nViews, (unsigned)(adjncy.size() / 2));
}

std::vector<Scene> SceneCluster::SplitSceneAggregativeClustering(std::vector<IIndexArr>* outLocalToGlobal)
{
	const IIndex nViews = scene.images.size();
	std::vector<IIndexArr> clusters(nViews);
	std::vector<int> nodeToCluster(nViews);
	for (IIndex i = 0; i < nViews; ++i) {
		clusters[i].push_back(i);
		nodeToCluster[i] = i;
	}

	struct Edge {
		int u, v;
		int weight;
		bool operator<(const Edge& other) const { return weight < other.weight; }
	};
	std::priority_queue<Edge> pq;
	std::vector<std::unordered_map<int, int>> adj(nViews);

	auto rebuildPQ = [&]() {
		pq = std::priority_queue<Edge>();
		for (size_t i = 0; i < clusters.size(); ++i) {
			adj[i].clear();
		}
		for (IIndex u = 0; u < nViews; ++u) {
			int cu = nodeToCluster[u];
			for (int i = xadj[u]; i < xadj[u+1]; ++i) {
				int v = adjncy[i];
				int cv = nodeToCluster[v];
				if (cu < cv) {
					adj[cu][cv] += adjwgt[i];
				}
			}
		}
		for (size_t i = 0; i < clusters.size(); ++i) {
			if (clusters[i].empty()) continue;
			for (const auto& p : adj[i]) {
				int target = p.first;
				int w = p.second;
				adj[target][i] = w;
				pq.push({(int)i, target, w});
			}
		}
	};

	rebuildPQ();

	unsigned numMergesSinceRefine = 0;
	while (!pq.empty()) {
		Edge e = pq.top();
		pq.pop();

		int u = e.u;
		int v = e.v;
		if (clusters[u].empty() || clusters[v].empty()) continue;
		if (adj[u].find(v) == adj[u].end() || adj[u][v] != e.weight) continue;

		if (clusters[u].size() + clusters[v].size() > config.maxViewsPerCluster) continue;

		for (IIndex node : clusters[v]) {
			nodeToCluster[node] = u;
		}
		for (IIndex val : clusters[v]) {
			clusters[u].push_back(val);
		}
		clusters[v].clear();
		numMergesSinceRefine++;

		adj[u].erase(v);
		for (const auto& p : adj[v]) {
			int nxt = p.first;
			int w = p.second;
			if (nxt == u) continue;
			adj[nxt].erase(v);
			adj[u][nxt] += w;
			adj[nxt][u] += w;
			pq.push({u, nxt, adj[u][nxt]});
		}
		adj[v].clear();

		const unsigned mergesPerRefine = MAXF(10u, config.maxViewsPerCluster / 10);
		if (numMergesSinceRefine >= mergesPerRefine) {
			RefineClustersLocalSearch(clusters);
			for (size_t c = 0; c < clusters.size(); ++c) {
				for (IIndex n : clusters[c]) {
					nodeToCluster[n] = (int)c;
				}
			}
			adj.resize(clusters.size());
			rebuildPQ();
			numMergesSinceRefine = 0;
		}
	}

	RefineClustersLocalSearch(clusters);
	MergeSmallClusters(clusters);
	RefineClustersSplitDisconnected(clusters);
	RefineClustersRescueOrphans(clusters);

	return BuildSubScenesFromClusters(clusters, outLocalToGlobal);
}

void SceneCluster::MergeSmallClusters(std::vector<IIndexArr>& clusters)
{
	const IIndex nViews = scene.images.size();
	std::vector<int> nodeToCluster(nViews, -1);
	for (size_t c = 0; c < clusters.size(); ++c) {
		for (IIndex u : clusters[c]) {
			nodeToCluster[u] = (int)c;
		}
	}

	bool changed = true;
	while (changed) {
		changed = false;
		for (size_t c = 0; c < clusters.size(); ++c) {
			if (clusters[c].size() == 0 || clusters[c].size() >= config.minViewsPerCluster) continue;

			std::unordered_map<int, int> cluster_weights;
			for (IIndex u : clusters[c]) {
				for (int i = xadj[u]; i < xadj[u+1]; ++i) {
					int v = adjncy[i];
					int target_c = nodeToCluster[v];
					if (target_c != (int)c && target_c != -1) {
						cluster_weights[target_c] += adjwgt[i];
					}
				}
			}

			int best_target = -1;
			int max_weight = -1;
			for (const auto& p : cluster_weights) {
				if (p.second > max_weight) {
					if (clusters[p.first].size() + clusters[c].size() <= config.maxViewsPerCluster + config.maxOverCapacity) {
						max_weight = p.second;
						best_target = p.first;
					}
				}
			}

			if (best_target != -1) {
				for (IIndex u : clusters[c]) {
					nodeToCluster[u] = best_target;
				}
				for (IIndex val : clusters[c]) {
					clusters[best_target].push_back(val);
				}
				clusters[c].clear();
				changed = true;
			}
		}
	}

	clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [](const IIndexArr& c) {
		return c.empty();
	}), clusters.end());
}

void SceneCluster::RefineClustersLocalSearch(std::vector<IIndexArr>& clusters)
{
	const IIndex nViews = scene.images.size();
	std::vector<int> nodeToCluster(nViews, -1);
	for (size_t c = 0; c < clusters.size(); ++c) {
		for (IIndex u : clusters[c]) {
			nodeToCluster[u] = (int)c;
		}
	}

	bool changed = true;
	int iters = 0;
	while (changed && iters < 20) {
		changed = false;
		iters++;
		for (IIndex u = 0; u < nViews; ++u) {
			int current_c = nodeToCluster[u];
			if (current_c == -1) continue;

			int best_target = current_c;
			int max_gain = 0;

			std::unordered_map<int, int> cluster_weights;
			for (int i = xadj[u]; i < xadj[u+1]; ++i) {
				int v = adjncy[i];
				int target_c = nodeToCluster[v];
				if (target_c != -1) {
					cluster_weights[target_c] += adjwgt[i];
				}
			}

			int current_internal_weight = cluster_weights[current_c];

			for (const auto& p : cluster_weights) {
				int target_c = p.first;
				int weight_to_target = p.second;
				if (target_c == current_c) continue;
				if (clusters[target_c].size() < config.maxViewsPerCluster) {
					int gain = weight_to_target - current_internal_weight;
					if (gain > max_gain) {
						max_gain = gain;
						best_target = target_c;
					}
				}
			}

			if (best_target != current_c) {
				nodeToCluster[u] = best_target;
				clusters[best_target].push_back(u);
				auto it = std::find(clusters[current_c].begin(), clusters[current_c].end(), u);
				if (it != clusters[current_c].end()) {
					*it = clusters[current_c].back();
					clusters[current_c].pop_back();
				}
				changed = true;
			}
		}
	}

	clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [](const IIndexArr& c) {
		return c.empty();
	}), clusters.end());
}

void SceneCluster::RefineClustersSplitDisconnected(std::vector<IIndexArr>& clusters)
{
	const IIndex nViews = scene.images.size();
	std::vector<int> nodeToCluster(nViews, -1);
	for (size_t c = 0; c < clusters.size(); ++c) {
		for (IIndex u : clusters[c]) {
			nodeToCluster[u] = (int)c;
		}
	}

	std::vector<IIndexArr> new_clusters;
	for (size_t c = 0; c < clusters.size(); ++c) {
		if (clusters[c].empty()) continue;

		std::unordered_set<IIndex> remaining(clusters[c].begin(), clusters[c].end());
		bool first = true;
		while (!remaining.empty()) {
			IIndex start_node = *remaining.begin();
			IIndexArr component;
			std::queue<IIndex> q;
			q.push(start_node);
			remaining.erase(start_node);
			while (!q.empty()) {
				IIndex u = q.front();
				q.pop();
				component.push_back(u);
				for (int i = xadj[u]; i < xadj[u+1]; ++i) {
					IIndex v = adjncy[i];
					if (nodeToCluster[v] == (int)c && remaining.count(v)) {
						q.push(v);
						remaining.erase(v);
					}
				}
			}
			if (first) {
				clusters[c] = component;
				first = false;
			} else {
				new_clusters.push_back(component);
			}
		}
	}
	if (!new_clusters.empty()) {
		clusters.insert(clusters.end(), new_clusters.begin(), new_clusters.end());
	}
}

void SceneCluster::RefineClustersRescueOrphans(std::vector<IIndexArr>& clusters)
{
	const IIndex nViews = scene.images.size();
	std::vector<int> nodeToCluster(nViews, -1);
	for (size_t c = 0; c < clusters.size(); ++c) {
		for (IIndex u : clusters[c]) {
			nodeToCluster[u] = (int)c;
		}
	}

	for (size_t c = 0; c < clusters.size(); ++c) {
		if (clusters[c].empty() || clusters[c].size() >= config.minViewsPerCluster) continue;

		// This cluster is still too small, try to reassign its nodes individually
		IIndexArr nodes = clusters[c];
		clusters[c].clear();
		for (IIndex u : nodes) {
			std::unordered_map<int, int> cluster_weights;
			for (int i = xadj[u]; i < xadj[u+1]; ++i) {
				int v = adjncy[i];
				int target_c = nodeToCluster[v];
				if (target_c != -1 && target_c != (int)c) {
					cluster_weights[target_c] += adjwgt[i];
				}
			}

			int best_target = -1;
			int max_weight = -1;
			for (const auto& p : cluster_weights) {
				if (p.second > max_weight) {
					if (clusters[p.first].size() < config.maxViewsPerCluster + config.maxOverCapacity) {
						max_weight = p.second;
						best_target = p.first;
					}
				}
			}

			if (best_target != -1) {
				nodeToCluster[u] = best_target;
				clusters[best_target].push_back(u);
			} else {
				// Could not find a better cluster, put it back (or leave it if we want to skip it)
				// For now, let's try to put it in any neighbor that has space, even if weight is 0
				for (size_t tc = 0; tc < clusters.size(); ++tc) {
					if (tc != c && !clusters[tc].empty() && clusters[tc].size() < config.maxViewsPerCluster + config.maxOverCapacity) {
						best_target = (int)tc;
						break;
					}
				}
				if (best_target != -1) {
					nodeToCluster[u] = best_target;
					clusters[best_target].push_back(u);
				} else {
					// No choice, put it back to its original (it will still be small/skipped)
					clusters[c].push_back(u);
				}
			}
		}
	}

	clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [](const IIndexArr& c) {
		return c.empty();
	}), clusters.end());
}

std::vector<Scene> SceneCluster::BuildSubScenesFromClusters(
	std::vector<IIndexArr>& clusters,
	std::vector<IIndexArr>* outLocalToGlobal)
{
	IIndex nSkippedViews = 0;
	std::vector<Scene> subScenes;
	subScenes.reserve(clusters.size());
	if (outLocalToGlobal)
		outLocalToGlobal->reserve(clusters.size());

	const unsigned nClusters = (unsigned)clusters.size();
	const unsigned nThreadsPerCluster = MAXF(1u, scene.nMaxThreads / MAXF(nClusters, 1u));
	DEBUG_EXTRA("Allocating %u threads per sub-scene (%u clusters, %u parent threads)",
		nThreadsPerCluster, nClusters, scene.nMaxThreads);

	for (IIndexArr& cluster : clusters) {
		// Sort by global ID so that local IDs preserve global ordering:
		// localID1 < localID2 means also globalID1 < globalID2
		// so pair ID ordering (ID1 < ID2) is maintained through local-global remapping
		cluster.Sort();
		if (cluster.size() < config.minViewsPerCluster) {
			DEBUG("warning: skipping small cluster with %u views", (unsigned)cluster.size());
			nSkippedViews += cluster.size();
			continue;
		}
		IIndexArr globalToLocal(scene.images.size());
		globalToLocal.MemsetValue(NO_ID);
		FOREACH(localID, cluster)
			globalToLocal[cluster[localID]] = localID;
		subScenes.emplace_back(ExtractSubScene(cluster, globalToLocal, nThreadsPerCluster));
		if (outLocalToGlobal)
			outLocalToGlobal->emplace_back(std::move(cluster));
	}
	DEBUG("Clustering: split into %u sub-scenes and %u skipped views, %u cross-sub-scene pairs remain",
		(unsigned)subScenes.size(), nSkippedViews, scene.pairs.size());
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2 && !subScenes.empty() && !subScenes[0].images.empty() && subScenes[0].images[0].View::metadata.HasGPS())
		ExportClusterPositions(subScenes, MAKE_PATH(String("clusters_gps.ply")));
	#endif
	return subScenes;
}

bool SceneCluster::ExportClusterPositions(
	const std::vector<Scene>& subScenes,
	const String& fileName)
{
	// Compute the ECEF centroid for normalizing the positions (optional, can help with visualization if large numbers)
	Point3dArr ecefPositions;
	Point3d centerECEF(0, 0, 0);
	FOREACH(clusterID, subScenes) {
		const Scene& scene = subScenes[clusterID];
		for (const Image& img : scene.images) {
			// Check if GPS data is valid (simple check: not all zero)
			const View::Metadata& viewMeta = img.View::metadata;
			if (!viewMeta.HasGPS())
				continue;
			Point3d ecef;
			WGS84ToECEF(viewMeta.latitude, viewMeta.longitude, viewMeta.altitude, ecef.x, ecef.y, ecef.z);
			ecefPositions.push_back(ecef);
			centerECEF += ecef;
		}
	}
	if (ecefPositions.empty()) {
		DEBUG("warning: no images with GPS positions found");
		return false;
	}
	centerECEF /= (double)ecefPositions.size();
	double lat0, lon0, alt0;
	ECEFToWGS84(centerECEF.x, centerECEF.y, centerECEF.z, lat0, lon0, alt0);

	// Define vertex structure for PLY export
	struct Vertex {
		Point3f p; // GPS position (longitude, latitude, altitude)
		Pixel8U c; // color (cluster ID)
	};
	// Define PLY properties
	static const PLY::PlyProperty props[] = {
		{"x",     PLY::Float32, PLY::Float32, offsetof(Vertex, p.x), 0, 0, 0, 0},
		{"y",     PLY::Float32, PLY::Float32, offsetof(Vertex, p.y), 0, 0, 0, 0},
		{"z",     PLY::Float32, PLY::Float32, offsetof(Vertex, p.z), 0, 0, 0, 0},
		{"red",   PLY::Uint8,   PLY::Uint8,   offsetof(Vertex, c.r), 0, 0, 0, 0},
		{"green", PLY::Uint8,   PLY::Uint8,   offsetof(Vertex, c.g), 0, 0, 0, 0},
		{"blue",  PLY::Uint8,   PLY::Uint8,   offsetof(Vertex, c.b), 0, 0, 0, 0}
	};
	// list of the kinds of elements in the PLY
	static const char* elem_names[] = {
		"vertex"
	};

	// Create PLY file
	PLY ply;
	if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE))
		return false;
	ply.describe_property("vertex", 6, props);
	ply.element_count("vertex", ecefPositions.size());
	if (!ply.header_complete())
		return false;

	// Generate unique color per cluster
	auto GenerateClusterColor = [](size_t clusterID, size_t numClusters) -> Pixel8U {
		if (numClusters == 1)
			return Pixel8U::RED; // red for single cluster
		// Generate distinct colors using HSV color space
		Pixel32F hsv{
			(float)clusterID / (float)numClusters * 360.f,
			0.9f,
			0.9f
		};
		Pixel32F rgb = CONVERT::HSV2RGB(hsv) * 255.f; // scale to [0, 255]
		return rgb.cast<uint8_t>();
	};

	// Write vertices
	unsigned vertexCount = 0;
	Vertex vertex;
	FOREACH(clusterID, subScenes) {
		const Scene& scene = subScenes[clusterID];
		const Pixel8U clusterColor = GenerateClusterColor(clusterID, subScenes.size());
		for (const Image& img : scene.images) {
			if (!img.View::metadata.HasGPS())
				continue;
			const Point3d& ecef = ecefPositions[vertexCount++];
			// Convert ECEF to ENU (centered at centroid)
			double e, n, u;
			ECEFToENU(ecef.x, ecef.y, ecef.z, centerECEF.x, centerECEF.y, centerECEF.z, lat0, lon0, e, n, u);
			// Store ENU position (east, north, up)
			vertex.p.x = static_cast<float>(e);
			vertex.p.y = static_cast<float>(n);
			vertex.p.z = static_cast<float>(u);
			vertex.c = clusterColor;
			ply.put_element(&vertex);
		}
	}

	VERBOSE("Exported %u GPS positions corresponding to %u clusters to '%s'",
		(unsigned)ecefPositions.size(), (unsigned)subScenes.size(), fileName.c_str());
	return true;
}
/*----------------------------------------------------------------*/
