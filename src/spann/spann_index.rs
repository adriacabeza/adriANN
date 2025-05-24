use std::io;

use crate::clustering::Cluster;
use crate::core::float::AdriannFloat;
use crate::distances::distance::DistanceMetric;
use crate::distances::SquaredEuclideanDistance;
use crate::spann::posting_lists::{FileBasedPostingListStore, PointData, PostingListStore};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use kiddo::KdTree;
use log::error;
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::fs::File;
use std::io::BufWriter;

pub struct SpannIndex<const N: usize, F: AdriannFloat> {
    pub kdtree: Option<KdTree<F, N>>,
    pub posting_list_store: Option<FileBasedPostingListStore>,
    posting_list_dir: String,
}

impl<const N: usize, F: AdriannFloat> SpannIndex<N, F> {
    pub fn new(posting_lists_dir: &str) -> io::Result<Self> {
        Ok(Self {
            kdtree: None,
            posting_list_store: None,
            posting_list_dir: posting_lists_dir.to_string(),
        })
    }

    pub fn load_posting_list(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        match PostingListStore::<F>::load_from_directory(path) {
            Ok(posting_list_store) => {
                self.posting_list_store = Some(posting_list_store);
                Ok(())
            }
            Err(e) => {
                error!("Failed to load posting list from {}: {}", path, e);
                Err(Box::new(e))
            }
        }
    }

    pub fn save_posting_list(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(posting_list_store) = &self.posting_list_store {
            PostingListStore::<F>::save_to_directory(posting_list_store)?;
            Ok(())
        } else {
            error!("Posting list is not available");
            Err("Posting list is not available".into())
        }
    }

    /// Create disk-backed posting list from the given clusters.
    pub fn create_posting_lists(
        &mut self,
        data: &ArrayView2<F>,
        clusters: &[Cluster],
    ) -> Option<&FileBasedPostingListStore> {
        let mut posting_list_store = FileBasedPostingListStore::new(&self.posting_list_dir)
            .expect("Failed to create posting list store");

        for (cluster_id, cluster) in clusters.iter().enumerate() {
            let num_points = cluster.points.len();
            let num_features = data.ncols();
            let mut points_array = Array2::zeros((num_points, num_features));
            let point_ids: Vec<usize> = cluster.points.clone();
            for (i, &point_idx) in cluster.points.iter().enumerate() {
                points_array.row_mut(i).assign(&data.row(point_idx));
            }
            if let Err(e) =
                posting_list_store.insert_posting_list(cluster_id, points_array, point_ids)
            {
                error!(
                    "Failed to insert posting list for cluster {}: {}",
                    cluster_id, e
                );
                return None;
            }
        }
        self.posting_list_store = Some(posting_list_store);
        self.posting_list_store.as_ref()
    }

    /// Builds the KD-tree from the centroids of the given `clusters`.
    /// Returns the resulting KdTree, also stored in self.
    pub fn build_kdtree(
        &mut self,
        data: &ArrayView2<F>,
        clusters: &[Cluster],
    ) -> Option<&KdTree<F, N>> {
        if data.ncols() != N {
            error!("Data column count does not match N={}", N);
            return None;
        }

        let mut tree: KdTree<F, N> = KdTree::new();
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            if let Some(centroid_idx) = cluster.centroid_idx {
                if let Some(array_slice) = data.row(centroid_idx).as_slice() {
                    if let Ok(array) = array_slice.try_into() {
                        tree.add(&array, cluster_id as u64);
                    } else {
                        error!("Centroid length mismatch for cluster {}", cluster_id);
                    }
                } else {
                    error!("Failed to extract centroid row for cluster {}", cluster_id);
                }
            }
        }
        self.kdtree = Some(tree);
        self.kdtree.as_ref()
    }

    pub fn save_kdtree(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = BufWriter::new(File::create(path)?);
        let encoder = GzEncoder::new(file, Compression::fast());
        if let Some(tree) = &self.kdtree {
            match bincode::serialize_into(encoder, tree) {
                Ok(_) => Ok(()),
                Err(e) => {
                    error!("Failed to serialize KD-Tree to {}: {}", path, e);
                    Err(Box::new(e))
                }
            }
        } else {
            error!("KD-Tree is not available");
            Err("KD-Tree is not available".into())
        }
    }

    pub fn load_kdtree(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let decompressor = GzDecoder::new(file);
        match bincode::deserialize_from(decompressor) {
            Ok(tree) => {
                self.kdtree = Some(tree);
                Ok(())
            }
            Err(e) => {
                error!("Failed to deserialize KD-Tree from {}: {}", path, e);
                Err(Box::new(e))
            }
        }
    }

    pub fn find_k_nearest_neighbor_spann(
        &self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> Option<Vec<PointData<F>>> {
        let tree = self.kdtree.as_ref().expect("KD-Tree is not available");
        let posting_list_store = self
            .posting_list_store
            .as_ref()
            .expect("Posting list is not available");

        let query_array: [F; N] = query
            .as_slice()
            .and_then(|slice| slice.try_into().ok())
            .expect("Query length mismatch");

        let nearest_centroids = tree.nearest_n::<kiddo::SquaredEuclidean>(&query_array, k);
        // Get the distance to the nearest centroid (c1)
        let nearest_centroid_dist = nearest_centroids[0].distance;
        // Set gamma to 0.2 as specified
        let gamma = F::from(0.2).unwrap();
        // Calculate the threshold: (1 + gamma) * Dist(q,c1)
        let threshold = (F::one() + gamma) * nearest_centroid_dist;

        let mut all_candidates: Vec<(F, PointData<F>)> = Vec::new();
        for nn in nearest_centroids {
            // Only search posting list if its centroid is within the threshold
            if nn.distance <= threshold {
                if let Ok(Some(points)) = posting_list_store.get_posting_list(nn.item as usize) {
                    for point_data in points {
                        let point_vector = ArrayView1::from(&point_data.vector);
                        let dist = SquaredEuclideanDistance.compute(&query.view(), &point_vector);
                        all_candidates.push((dist, point_data));
                    }
                }
            }
        }

        if all_candidates.is_empty() {
            error!("No candidate points found for nearest centroids.");
            return None;
        }

        all_candidates
            .sort_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal));

        if all_candidates.len() > k {
            all_candidates.truncate(k);
        }
        let nearest_points = all_candidates.into_iter().map(|(_, point)| point).collect();

        Some(nearest_points)
    }
}
