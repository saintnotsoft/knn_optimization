#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h> // For DBL_MAX
#include <string.h> // For memcpy

// --- 1. CONFIGURATION AND CORE STRUCTS ---

#define D 2             // Dimensionality
#define LEAF_SIZE 10    // Max points in a leaf node

// A. Data Point
typedef struct {
    double vector[D];
    int id;
} DataPoint;

// B. Ball Tree Node
typedef struct BallTreeNode {
    double centroid[D]; // Center of the ball
    double radius;      // Radius of the ball

    // Leaf nodes store points, internal nodes store children
    DataPoint *points;  // Array of points (NULL for internal nodes)
    int num_points;     // Number of points in this leaf
    
    struct BallTreeNode *left;
    struct BallTreeNode *right;
} BallTreeNode;

// C. Neighbor (for storing results)
typedef struct {
    int id;
    double distance;
} Neighbor;

// D. Simplified KNN Result Holder (Replaces Max-Heap)
typedef struct {
    Neighbor *neighbors; // A sorted array of the K best neighbors
    int K;
    int count;           // How many neighbors we've found so far (up to K)
} KNNResult;

// --- 2. DISTANCE FUNCTION ---

// Simple Euclidean Distance
double euclidean_distance(const double *v1, const double *v2) {
    double sum_sq = 0.0;
    for (int i = 0; i < D; i++) {
        double diff = v1[i] - v2[i];
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq);
}

// --- 3. SIMPLIFIED KNN RESULT MANAGEMENT ---

// Create the result holder
KNNResult* create_knn_result(int K) {
    KNNResult *res = (KNNResult*)malloc(sizeof(KNNResult));
    res->K = K;
    res->count = 0;
    res->neighbors = (Neighbor*)malloc(sizeof(Neighbor) * K);
    
    // Initialize all distances to "infinity"
    for (int i = 0; i < K; i++) {
        res->neighbors[i].distance = DBL_MAX;
        res->neighbors[i].id = -1;
    }
    return res;
}

// Insert a new neighbor into the sorted list
void insert_neighbor(KNNResult *res, Neighbor n) {
    // Check if this neighbor is worse than the worst neighbor we already have
    if (res->count == res->K && n.distance >= res->neighbors[res->K - 1].distance) {
        return;
    }

    // Find the correct insertion spot
    int i = res->count;
    if (i > res->K - 1) i = res->K - 1; // Start at the end

    // Shift elements down to make room
    while (i > 0 && n.distance < res->neighbors[i - 1].distance) {
        res->neighbors[i] = res->neighbors[i - 1];
        i--;
    }
    
    res->neighbors[i] = n;
    
    if (res->count < res->K) {
        res->count++;
    }
}

void free_knn_result(KNNResult *res) {
    free(res->neighbors);
    free(res);
}

// --- 4. BALL TREE CONSTRUCTION (Simplified Split) ---

// Helper: Compute centroid and radius for a set of points
void compute_ball_params(BallTreeNode *node, DataPoint *points, int n) {
    if (n == 0) return;

    // 1. Compute Centroid (Mean)
    for (int j = 0; j < D; j++) node->centroid[j] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < D; j++) {
            node->centroid[j] += points[i].vector[j];
        }
    }
    for (int j = 0; j < D; j++) node->centroid[j] /= n;

    // 2. Compute Radius (Max distance from centroid)
    node->radius = 0.0;
    for (int i = 0; i < n; i++) {
        double d = euclidean_distance(points[i].vector, node->centroid);
        if (d > node->radius) {
            node->radius = d;
        }
    }
}

// Global variable to help qsort
int split_dim_global; 
// qsort comparison function
int compare_points(const void *a, const void *b) {
    DataPoint *pA = (DataPoint*)a;
    DataPoint *pB = (DataPoint*)b;
    if (pA->vector[split_dim_global] < pB->vector[split_dim_global]) return -1;
    if (pA->vector[split_dim_global] > pB->vector[split_dim_global]) return 1;
    return 0;
}

// Recursive build function
BallTreeNode* build_ball_tree(DataPoint *points, int n) {
    if (n == 0) return NULL;

    BallTreeNode *node = (BallTreeNode*)malloc(sizeof(BallTreeNode));
    node->left = NULL;
    node->right = NULL;

    // Compute this node's bounding ball *before* splitting
    compute_ball_params(node, points, n);

    // Base Case: Leaf Node
    if (n <= LEAF_SIZE) {
        node->num_points = n;
        // Leaf node stores a *copy* of its points
        node->points = (DataPoint*)malloc(sizeof(DataPoint) * n);
        memcpy(node->points, points, sizeof(DataPoint) * n);
        return node;
    }

    // --- Internal Node: Find Split ---

    // 1. Find dimension with the largest variance (spread)
    int split_dim = 0;
    double max_spread = -1.0;
    for (int d = 0; d < D; d++) {
        double min_val = DBL_MAX, max_val = -DBL_MAX;
        for (int i = 0; i < n; i++) {
            if (points[i].vector[d] < min_val) min_val = points[i].vector[d];
            if (points[i].vector[d] > max_val) max_val = points[i].vector[d];
        }
        double spread = max_val - min_val;
        if (spread > max_spread) {
            max_spread = spread;
            split_dim = d;
        }
    }

    // 2. Sort points along that dimension to find the median
    split_dim_global = split_dim; // Use global for qsort
    qsort(points, n, sizeof(DataPoint), compare_points);

    // 3. Find median index
    int median_idx = n / 2;
    
    // 4. Copy points into left/right arrays
    int n_left = median_idx;
    int n_right = n - median_idx;
    
    DataPoint *left_points = (DataPoint*)malloc(sizeof(DataPoint) * n_left);
    DataPoint *right_points = (DataPoint*)malloc(sizeof(DataPoint) * n_right);
    
    memcpy(left_points, points, sizeof(DataPoint) * n_left);
    memcpy(right_points, points + median_idx, sizeof(DataPoint) * n_right);

    // Internal node doesn't store points
    node->points = NULL;
    node->num_points = 0;
    
    // 5. Recurse
    node->left = build_ball_tree(left_points, n_left);
    node->right = build_ball_tree(right_points, n_right);
    
    // Free the temporary arrays (data is copied into children)
    free(left_points);
    free(right_points);
    
    return node;
}


// --- 5. OPTIMIZED KNN QUERY ---

// Distance from query point to the *surface* of a ball
double min_distance_to_ball(const double *query, const BallTreeNode *node) {
    double dist_to_center = euclidean_distance(query, node->centroid);
    
    if (dist_to_center <= node->radius) {
        // Query point is inside the ball
        return 0.0;
    }
    
    // Distance = (dist to center) - radius
    return dist_to_center - node->radius;
}

// Recursive search function
void ball_tree_knn_query(BallTreeNode *node, const double *query, int K, KNNResult *result) {
    if (node == NULL) return;

    // Get the distance of the K-th neighbor (the "worst" one we've found)
    double farthest_neighbor_dist = (result->count < K) ? DBL_MAX : result->neighbors[K - 1].distance;

    // --- PRUNING OPTIMIZATION ---
    // Calculate shortest possible distance from query to this ball
    double min_dist_to_node = min_distance_to_ball(query, node);

    // If the closest point in this ball is farther than our K-th neighbor,
    // we can skip (prune) this entire branch.
    if (min_dist_to_node >= farthest_neighbor_dist) {
        return;
    }

    // --- Leaf Node: Brute-force check ---
    if (node->points != NULL) { // This is a leaf node
        for (int i = 0; i < node->num_points; i++) {
            double d = euclidean_distance(query, node->points[i].vector);
            Neighbor n = {.id = node->points[i].id, .distance = d};
            insert_neighbor(result, n);
        }
        return;
    }

    // --- Internal Node: Recurse ---
    
    // Prioritize searching the child node whose center is closer
    double dist_left = euclidean_distance(query, node->left->centroid);
    double dist_right = euclidean_distance(query, node->right->centroid);

    if (dist_left < dist_right) {
        // Search closer child (left) first
        ball_tree_knn_query(node->left, query, K, result);
        // Search farther child (right) second
        ball_tree_knn_query(node->right, query, K, result);
    } else {
        // Search closer child (right) first
        ball_tree_knn_query(node->right, query, K, result);
        // Search farther child (left) second
        ball_tree_knn_query(node->left, query, K, result);
    }
}

// --- 6. MEMORY MANAGEMENT AND MAIN ---

void free_ball_tree(BallTreeNode *node) {
    if (node == NULL) return;
    
    // Free children first
    free_ball_tree(node->left);
    free_ball_tree(node->right);
    
    // Free points array if it's a leaf
    if (node->points != NULL) {
        free(node->points);
    }
    
    // Free the node itself
    free(node);
}

int main() {
    int N = 5; // Number of data points
    int K = 2; // Number of neighbors to find

    // Create Sample Data (a single contiguous array)
    DataPoint *data = (DataPoint*)malloc(sizeof(DataPoint) * N);
    
    data[0] = (DataPoint){{1.0, 1.0}, 1};
    data[1] = (DataPoint){{5.0, 5.0}, 2};
    data[2] = (DataPoint){{2.0, 2.0}, 3};
    data[3] = (DataPoint){{10.0, 1.0}, 4};
    data[4] = (DataPoint){{1.0, 10.0}, 5};

    printf("Building Ball Tree...\n");
    // Pass the array and its size to the build function
    BallTreeNode *root = build_ball_tree(data, N);
    printf("Ball Tree built successfully.\n\n");

    // --- PERFORM KNN QUERY ---
    double query_point[D] = {1.5, 1.5};
    printf("Query Point: (%.1f, %.1f) - Finding K=%d nearest neighbors.\n", 
           query_point[0], query_point[1], K);

    KNNResult *knn_result = create_knn_result(K);
    
    ball_tree_knn_query(root, query_point, K, knn_result);

    // --- OUTPUT RESULTS ---
    printf("\n--- Results ---\n");
    
    for (int i = 0; i < knn_result->count; i++) {
        double dist = knn_result->neighbors[i].distance;
        int id = knn_result->neighbors[i].id;
        
        // Find the original point's coordinates (for display)
        double *p_vec = NULL;
        for(int j = 0; j < N; j++) {
            if (data[j].id == id) p_vec = data[j].vector;
        }
        
        printf("%d. ID %d at (%.1f, %.1f), Distance: %.4f\n", 
               i + 1, id, p_vec[0], p_vec[1], dist);
    }

    // --- CLEAN UP ---
    free_knn_result(knn_result);
    free_ball_tree(root);
    free(data);
    
    return 0;
}