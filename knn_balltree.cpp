/*
     DS Innovative Assignment
    KNN Optimization using Ball Tree Structure
    By: BAM 040 & 050


    KNN Optimization of the time complexity from O(n) to O(log n) using hyper-sphere tree structure and triangle inequality pruning.
    Goal: Implementing both Brute Force and Ball Tree Methods and comparing them, finding that Ball Tree method is much faster.
 */


#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <limits>  // added for numeric_limits

using namespace std;
using namespace chrono;


/*
    Point Class, allows the code to work with points of any dimension without changing the logic.
    using std::vector<double>
*/


class Point {
public:
    vector<double> coords;  // for storing coords
    int id;
    
    Point(int dimensions = 3, int identifier = -1) // defualt paramenters, 
        : coords(dimensions, 0.0), id(identifier) {} // coords vector filled with {0.0,0.0,0.0}
    
    Point(const vector<double>& c, int identifier = -1) // create from existing data
        : coords(c), id(identifier) {}
    
    double operator[](int i) const { return coords[i]; } // access the coords as if the point obj were an array
    double& operator[](int i) { return coords[i]; } // returns a reference, lets you change the coordinates
    int size() const { return coords.size(); }
};


/*
    function to calc euclidean distance btw two const objs
*/


double euclideanDistance(const Point& p1, const Point& p2) {
    double sum = 0.0;
    for (int i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


/*
    ball tree node struct
*/


struct BallTreeNode {
    Point center;                     // centre of hyper sphere
    double radius;                   // dist from centre to the farthest point
    vector<Point> points;            // if isLeaf is TRUE, stores small collection of actual data pointss that belongs to tis cluster
    unique_ptr<BallTreeNode> left;          // pointer to left child
    unique_ptr<BallTreeNode> right;         // pointer to right child
    bool isLeaf;                     // indicating whether a leaf node or not
    
    BallTreeNode() : radius(0.0), isLeaf(true) {}
};


/*
    priority queue for maintaing knn
*/


struct Neighbor {
    Point point;
    double distance;
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;  // max heap (bt where each parent node value if <= to its children node value)
    }
};


/*
    ball tree implementation
*/


class BallTree {
private:
    unique_ptr<BallTreeNode> root; // stores the root of entire tree
    int leafSize; // threshold, stopping tree recursion, if few points, becomes a leaf
    mutable long long distanceComputations; // total no of dist calculations, mutable becuase allows it to be changes inside const methods
    
    // find centroid of points
    Point computeCentroid(const vector<Point>& points) {
        Point centroid(points[0].size()); // ensuring correct dimensionality
        for (const auto& p : points) { // auto reduces the need for explicit type declarations
            for (int i = 0; i < p.size(); ++i) {
                centroid[i] += p[i]; // accumulate 
            }
        }
        for (int i = 0; i < centroid.size(); ++i) {
            centroid[i] /= points.size(); // avging step, sum of each dimension/total no of points
        }
        return centroid;
    }
    
    // find farthest point from given centroid
    int findFarthest(const Point& center, const vector<Point>& points) {
        double maxDist = -1.0; // start at negative value to ensure the distance to first point will always be greater
        int farthestIdx = 0; // start at index 0
        for (size_t i = 0; i < points.size(); ++i) {
            double dist = euclideanDistance(center, points[i]);
            if (dist > maxDist) {
                maxDist = dist;
                farthestIdx = i;
            }
        }
        return farthestIdx;
    }
    
    // recursion to built tree
    unique_ptr<BallTreeNode> buildTree(vector<Point>& points) {
        auto node = make_unique<BallTreeNode>();
        
        if (points.empty()) return node; // if no points, returns empty node
        
        // defining the bounding balls
        node->center = computeCentroid(points);
        node->radius = 0.0;
        for (const auto& p : points) {
            double dist = euclideanDistance(node->center, p);
            if (dist > node->radius) node->radius = dist;
        }
        
        // no further subdivision, stops recursion, threshold 
        if (points.size() <= leafSize) {
            node->isLeaf = true;
            node->points = points;
            return node;
        }
        
        // dicide current set of points into two subsets
        node->isLeaf = false;
        int pivot1Idx = findFarthest(node->center, points);
        Point pivot1 = points[pivot1Idx]; // first pivot farthest from centroid
        int pivot2Idx = findFarthest(pivot1, points);
        Point pivot2 = points[pivot2Idx]; // second pivot farthest from first pivot
        
        // Partition points
        vector<Point> leftPoints, rightPoints;
        for (const auto& p : points) {
            double dist1 = euclideanDistance(p, pivot1);
            double dist2 = euclideanDistance(p, pivot2);
            if (dist1 <= dist2) {
                leftPoints.push_back(p); // if close or equidistant from pivot 1, added to left points
            } else {
                rightPoints.push_back(p); // else added to right points
            }
        }
        
        // handle degenerate case
        if (leftPoints.empty() || rightPoints.empty()) {
            node->isLeaf = true; 
            node->points = points;
            return node;
        } // if one of the subset is empty, split failed to create two balanced child node, in thsi case node is forced to leaf  node, preventing infinite recursion
        
        // creates left and right child nodes till leaf size condition is met everywhere
        node->left = buildTree(leftPoints);
        node->right = buildTree(rightPoints);
        
        return node;
    }
    
    // search for knn
    void searchKNN(BallTreeNode* node, const Point& query, int k,
                   priority_queue<Neighbor>& neighbors) const {
        if (!node) return;
        
        // dist to sphere centre
        double centerDist = euclideanDistance(query, node->center);
        distanceComputations++; // track efficiency of code
        
        // triangle inequality pruning, if outside the sphere mindist is the dist from the query to the centroid, if inside the sphere mindist is 0.0
        double minDist = max(0.0, centerDist - node->radius);
        if (neighbors.size() == k && minDist >= neighbors.top().distance) {
            return;  // prune this subtree
        }
        
        // if node is a leaf, 
        if (node->isLeaf) {
            for (const auto& p : node->points) {
                double dist = euclideanDistance(query, p);
                distanceComputations++;
                
                if (neighbors.size() < k) { // queue isnt full i.e no of neighbours are less than the target no
                    neighbors.push({p, dist});
                } else if (dist < neighbors.top().distance) { // all the neighbours are sought, if new points dist is less than farthest current neighbour, that neighbour is popped and new closer neighbour is added
                    neighbors.pop();
                    neighbors.push({p, dist});
                }
            }
            return;
        }
        
        // visiting children nodes with safe null checks to fix errors
        double leftDist = node->left ? euclideanDistance(query, node->left->center) : std::numeric_limits<double>::max();
        double rightDist = node->right ? euclideanDistance(query, node->right->center) : std::numeric_limits<double>::max();
        distanceComputations += (node->left ? 1 : 0) + (node->right ? 1 : 0);
        
        if (leftDist < rightDist) { // searching child whose centre is closer to the query point
            if (node->left) searchKNN(node->left.get(), query, k, neighbors);
            if (node->right) searchKNN(node->right.get(), query, k, neighbors);
        } else {
            if (node->right) searchKNN(node->right.get(), query, k, neighbors);
            if (node->left) searchKNN(node->left.get(), query, k, neighbors);
        }
    }
    
public:
    BallTree(int leafSize = 40) : leafSize(leafSize), distanceComputations(0) {} // threshold contructor, default value is 40
    
    void build(vector<Point>& points) { // constructing ball tree using provided set of points
        distanceComputations = 0;
        root = buildTree(points);
    }
    
    vector<pair<Point, double>> findKNN(const Point& query, int k) const { // knn search for given query point, seeking k neighbours
        distanceComputations = 0;
        priority_queue<Neighbor> neighbors; // creating empty max heap priority queue
        searchKNN(root.get(), query, k, neighbors);
        
        vector<pair<Point, double>> results;
        while (!neighbors.empty()) {
            results.push_back({neighbors.top().point, neighbors.top().distance});
            neighbors.pop();
        }
        reverse(results.begin(), results.end()); // max-heap gives largest element first, reverse- closest to farthest
        return results;
    }
    
    long long getDistanceComputations() const { return distanceComputations; } // efficiency metric - retrieve total num of dist calc performed (N x D computations )
};


/*
    bfs for comparision
*/


class BruteForceKNN {
private:
    vector<Point> points;
    mutable long long distanceComputations;
    
public:
    BruteForceKNN() : distanceComputations(0) {} // initialize counter to 0
    
    void build(const vector<Point>& dataPoints) {
        points = dataPoints;
        distanceComputations = 0; // stores all the data points internally
    }
    
    vector<pair<Point, double>> findKNN(const Point& query, int k) const {
        distanceComputations = 0;
        priority_queue<Neighbor> neighbors; // initializing priority queue
        
        // iterate through every point
        for (const auto& p : points) {
            double dist = euclideanDistance(query, p); // calc dist btw q and p
            distanceComputations++; // track of dist calc's
            
            if (neighbors.size() < k) {
                neighbors.push({p, dist});
            } else if (dist < neighbors.top().distance) {
                neighbors.pop();
                neighbors.push({p, dist}); // new closer neighbour added
            }
        }
        
        vector<pair<Point, double>> results;
        while (!neighbors.empty()) {
            results.push_back({neighbors.top().point, neighbors.top().distance});
            neighbors.pop();
        }
        reverse(results.begin(), results.end());
        return results;
    }
    
    long long getDistanceComputations() const { return distanceComputations; }
};


/*
    data generation
*/


vector<Point> generateRandomPoints(int numPoints, int dimensions, int seed = 42) {
    mt19937 rng(seed); // initialize mersenne twister engine, using seed (defualting to 42), ensues sequence of random numbers are reproducible
    uniform_real_distribution<double> dist(0.0, 100.0); // uniform distribution, evenly distributed between 0.0 to 100.0, means all coordinates are covered
    
    vector<Point> points;
    for (int i = 0; i < numPoints; ++i) {
        Point p(dimensions, i);
        for (int d = 0; d < dimensions; ++d) {
            p[d] = dist(rng);
        }
        points.push_back(p);
    }
    return points;
}


/*
 benchmark and comparision
*/


void runBenchmark(int numPoints, int dimensions, int k, int numQueries) {
    cout << "\n" << string(80, '=') << endl;
    cout << "BENCHMARK: " << numPoints << " points, " << dimensions 
         << "D, K=" << k << ", Queries=" << numQueries << endl;
    cout << string(80, '=') << endl;
    
    // Generate data
    cout << "\nGenerating data..." << flush;
    auto dataPoints = generateRandomPoints(numPoints, dimensions);
    auto queries = generateRandomPoints(numQueries, dimensions, 100);
    cout << " Done!" << endl;
    
    // Build Ball Tree
    cout << "Building Ball Tree..." << flush;
    auto buildStart = high_resolution_clock::now();
    BallTree ballTree(40);
    ballTree.build(dataPoints);
    auto buildEnd = high_resolution_clock::now();
    double buildTime = duration_cast<milliseconds>(buildEnd - buildStart).count();
    cout << " Done! (" << buildTime << " ms)" << endl;
    
    // Build Brute Force
    cout << "Building Brute Force..." << flush;
    BruteForceKNN bruteForce;
    bruteForce.build(dataPoints);
    cout << " Done!" << endl;
    
    // Query Ball Tree
    cout << "\nQuerying Ball Tree..." << flush;
    auto ballStart = high_resolution_clock::now();
    long long totalBallDist = 0;
    for (const auto& query : queries) {
        auto neighbors = ballTree.findKNN(query, k);
        totalBallDist += ballTree.getDistanceComputations();
    }
    auto ballEnd = high_resolution_clock::now();
    double ballTime = duration_cast<microseconds>(ballEnd - ballStart).count() / 1000.0;
    cout << " Done! (" << ballTime << " ms)" << endl;
    
    // Query Brute Force
    cout << "Querying Brute Force..." << flush;
    auto bruteStart = high_resolution_clock::now();
    long long totalBruteDist = 0;
    for (const auto& query : queries) {
        auto neighbors = bruteForce.findKNN(query, k);
        totalBruteDist += bruteForce.getDistanceComputations();
    }
    auto bruteEnd = high_resolution_clock::now();
    double bruteTime = duration_cast<microseconds>(bruteEnd - bruteStart).count() / 1000.0;
    cout << " Done! (" << bruteTime << " ms)" << endl;
    
    // Results
    double speedup = bruteTime / ballTime;
    long long avgBallDist = totalBallDist / numQueries;
    long long avgBruteDist = totalBruteDist / numQueries;
    double pruning = 100.0 * (1.0 - (double)avgBallDist / avgBruteDist);
    
    cout << "\n" << string(80, '-') << endl;
    cout << "RESULTS:" << endl;
    cout << string(80, '-') << endl;
    cout << fixed << setprecision(2);
    cout << "Ball Tree Build Time:      " << buildTime << " ms" << endl;
    cout << "Ball Tree Query Time:      " << ballTime << " ms" << endl;
    cout << "Brute Force Query Time:    " << bruteTime << " ms" << endl;
    cout << "Speedup:                  " << speedup << "x" << endl;
    cout << "\nDistance Computations (avg per query):" << endl;
    cout << "  Ball Tree:               " << avgBallDist << endl;
    cout << "  Brute Force:             " << avgBruteDist << endl;
    cout << "  Pruning Efficiency:      " << pruning << "%" << endl;
    cout << string(80, '=') << endl;
}


/*
    verification
*/


void verifyCorrectness() {
    cout << "\n" << string(80, '=') << endl;
    cout << "CORRECTNESS VERIFICATION" << endl;
    cout << string(80, '=') << endl;
    
    auto dataPoints = generateRandomPoints(500, 3); // generates 500 data points in 3D
    auto query = generateRandomPoints(1, 3, 999)[0]; // generates 1 query point
    int k = 5; // test it with 5 nearest neighbours
    
    BallTree ballTree;
    ballTree.build(dataPoints); 
    
    BruteForceKNN bruteForce;
    bruteForce.build(dataPoints);
    
    auto ballResults = ballTree.findKNN(query, k);
    auto bruteResults = bruteForce.findKNN(query, k);
    
    cout << "\nComparing top " << k << " neighbors:" << endl;
    bool correct = true;
    for (int i = 0; i < k; ++i) {
        double diff = abs(ballResults[i].second - bruteResults[i].second); // diff between dist values {point,dist} pairs
        cout << "  Rank " << (i+1) << ": Ball Tree=" << fixed << setprecision(6)
             << ballResults[i].second << ", Brute Force=" << bruteResults[i].second;
        if (diff > 1e-9) { // tolerance set to a billionth 
            cout << " ✗ MISMATCH"; 
            correct = false;
        } else {
            cout << " ✓";
        }
        cout << endl;
    }
    
    cout << "\n";
    if (correct) {
        cout << "✓ VERIFICATION PASSED: Results match!" << endl;
    } else {
        cout << "✗ VERIFICATION FAILED: Results differ!" << endl;
    }
    cout << string(80, '=') << endl;
}



int main() {
    cout << "\n";
    cout << "============================================================================\n";
    cout << "                                                                          \n";
    cout << "           K-NEAREST NEIGHBORS OPTIMIZATION USING BALL TREE                 \n";
    cout << "                                                                          \n";
    cout << "  Demonstrating O(log N) query complexity vs O(N) brute-force approach     \n";
    cout << "                                                                          \n";
    cout << "============================================================================\n";
    
    // Verify correctness
    verifyCorrectness();
    
    // Run benchmarks
    runBenchmark(1000, 3, 5, 100);
    runBenchmark(5000, 5, 10, 100);
    runBenchmark(10000, 10, 15, 50);
    
    cout << "\n✓ All tests completed successfully!\n" << endl;
    
    return 0;
}
