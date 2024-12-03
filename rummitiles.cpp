// Online C++ compiler to run C++ program online
//#include "function_pool.h"
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>   
#include <mutex>
#include <functional>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <future>

// Custom Thread Pool class (same as previous example)
class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    // Submit a task to the thread pool
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // Worker threads
    void worker();

    // Task queue and synchronization primitives
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> workers;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.push_back(std::thread(&ThreadPool::worker, this));
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stop = true;
    }

    condition.notify_all();  // Wake up all workers
    for (auto& worker : workers) {
        worker.join();
    }
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            // Wait for a task or a stop signal
            condition.wait(lock, [this] { return stop || !tasks.empty(); });

            if (stop && tasks.empty()) {
                return;
            }

            task = std::move(tasks.front());
            tasks.pop();
        }

        // Execute the task
        task();
    }
}

template <typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    // Wrap the function into a packaged task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (stop) {
            throw std::runtime_error("ThreadPool is stopped, cannot submit new tasks");
        }

        tasks.push([task]() { (*task)(); });
    }

    condition.notify_one();  // Notify one worker thread

    return res;
}

class Tile {
    public:
    int color;
    int value;

    int x;
    int y;
    
    Tile(int o_value, int o_color, int o_x, int o_y){
        value = o_value;
        color = o_color;
        x = o_x;
        y = o_y;
    }

    std::ostream& operator<<(std::ostream& os);

    friend std::ostream& operator<<(std::ostream& os, const Tile& t){
        os << "[value: " << t.value << " color: " << t.color << "]";
        return os;
    }

    bool isWild(){
        return value == -1 && color == -1;
    }

};

void printTileVec(std::vector<Tile *>& c) {
    for (int i = 0; i < c.size(); i++){
        std::cout << *c[i] << " ";
    }
    std::cout << std::endl;
}

void printBoard(std::vector<std::vector<Tile *> >& b) {
    for (int i = 0; i < b.size(); i++){
        std::cout << "row " << i << ": ";
        printTileVec(b[i]);
    }
    std::cout << std::endl;
}
        
std::vector<Tile *> pretty_print(std::vector<Tile *>& c, long long int combo, int starting_tile_size)
{
    long long int n = c.size();
    std::vector<Tile *> combination;
    combination.reserve(starting_tile_size);
    for (long long int i = 0; i < n; ++i) {
        if ((combo >> i) & 1) {
            combination.push_back(c[i]);
        }
    }
    return combination;
}

std::vector<std::vector<Tile *> > combo(std::vector<Tile *>& c, int starting_tile_size)
{
    std::vector<std::vector<Tile *> > ret;
    long long int n = c.size();
    long long int combo = (1 << (long long int) starting_tile_size) - 1; // starting_tile_size bit sets
    while (combo < (long long int)1<<n) {

        std::vector<Tile *> combination = pretty_print(c, combo, starting_tile_size);

        long long int x = combo & -combo;
        long long int y = combo + x;
        long long int z = (combo & ~y);
        combo = z / x;
        combo >>= 1;
        combo |= y;
        ret.push_back(combination);
    }
    return ret;
}

std::vector<std::vector<Tile*> > getSets(std::vector<Tile *> &startingCombination, int range){
    std::vector<Tile*> tempVec;
    std::vector<std::vector<Tile*> > ret(range, tempVec);

    for (unsigned int i = 0; i < startingCombination.size(); i++){
        int val = startingCombination[i]->value;
        if(ret[val].empty()){
            std::vector<Tile *> vec;
            vec.push_back(startingCombination[i]);
            ret[val] = vec;
        }
        else{
            ret[val].push_back(startingCombination[i]);
        }
    }

    // for determinism, sort all 2nd level vectors
    for (unsigned int i = 0; i < ret.size(); i++){
        std::sort(ret[i].begin(), ret[i].end());
    }

    return ret;
}

// we assume the tiles are all adjacent
bool isSet(std::vector<Tile *> &Tiles){
    if (Tiles.size() < 3){
        return false;
    }
    int val = Tiles[0]->isWild() ? Tiles[1]->value : Tiles[0]->value;
    for (Tile * t : Tiles){
        if (t->value != val && !t->isWild()){
            return false;
        }
    }
    return true;
}

// we assume the tiles are all adjacent
// this is only called from isValid, so we don't need
// to sort the tiles by value
bool isRun(std::vector<Tile *> &Tiles, int range){
    if (Tiles.size() < 3){
        return false;
    }

    int color = Tiles[0]->isWild() ? Tiles[1]->color : Tiles[0]->color;

    // first do a color check
    for (int i = 0; i < Tiles.size()-2; i++){
        Tile *t1 = Tiles[i];
        
        if (t1->color != color && !t1->isWild()){
            return false;
        }
    }

    // now do a value check
    for (int i = 0; i < Tiles.size()-2; i++){
        Tile *t1 = Tiles[i];
        Tile *t2 = Tiles[i+1];
        Tile *t3 = Tiles[i+2];
        int nextValue = t1->value == range-1 ? 0 : t1->value + 1;
        int nextNextValue = nextValue == range-1 ? 0 : nextValue + 1;

        if (t1->isWild()){
            if (t2->value != nextValue){
                return false;
            }
            if (t3->value != nextNextValue){
                return false;
            }
        }
        else{            
            if (t2->value != nextValue && !t2->isWild()){
                return false;
            }
            if (t3->value != nextNextValue && !t3->isWild()){
                return false;
            }
        }
    }
    return true;
}

// Custom comparator for A type
bool comp(Tile* x, Tile *y) {
    return x->value < y->value;
}

// returns a data structure of all the runs contained in the starting combination. No subsequences. 
// It's a 3D vector, for any given value and color, the structure will contain the longest ascending run that starts
// with that tile. It will not contain any entry if there's an earlier start tile.
// e.g. a run of color 0 for values 2, 3, 4, will not have an entry for value: 3 color: 0.
// That run will only be referenced at value: 2 color: 0
std::vector<std::vector<std::vector< Tile *> > > getRuns(std::vector<Tile *> startingCombination, int range, int colors){
    std::vector<std::vector<std::vector<Tile *> > > ret(range, std::vector<std::vector<Tile *> >(colors, std::vector<Tile *>(0)));

    // sort by ascending value
    std::sort(startingCombination.begin(), startingCombination.end(), comp);
    
    int index = 0;
    while (startingCombination.size() != 0){
        // always find the run associated with the first value
        int tileVal = startingCombination[index]->value;        
        int tileColor = startingCombination[index]->color;        
        int lowerVal = tileVal == 0 ? range-1 : tileVal - 1;
       
        int max_iteration = startingCombination.size();
        while (true){
            bool found = false;
            for (int i = startingCombination.size()-1; i >= 0; i--){
                if (startingCombination[i]->value == lowerVal && 
                startingCombination[i]->color == tileColor){
                    found = true;
                    lowerVal = lowerVal == 0 ? range-1 : lowerVal - 1;
                    tileVal = lowerVal;
                    max_iteration--;
                    index = i;
                    break;
                }
            }
            if (max_iteration == 0 || !found){
                break;
            }
        }        

        // now we are on the "lowest tile", so we start building the run.
        std::vector<Tile *> run;
        int originalSize = startingCombination.size();
        for (unsigned int i = 0; i < originalSize; i++){
            // add the known good tile to the run, remove it from the starting combo,
            // and locate the next value we're searching for
            int tempIndex = (index + i) % startingCombination.size();
            int nextValue = startingCombination[tempIndex]->value == range-1 ? 0 : startingCombination[tempIndex]->value + 1;
            run.push_back(startingCombination[tempIndex]);            
            startingCombination.erase(startingCombination.begin() + tempIndex);

            // an empty starting combo means we're done
            if (startingCombination.size() == 0){
                break;
            }

            // search the whole starting combo for the next value
            // if we find it, set up for the next iteration
            bool found = false;
            for (int j = 0; j < startingCombination.size(); j++){
                if (startingCombination[j]->value == nextValue && startingCombination[j]->color == tileColor){
                    index = j-i-1; // index is added to i, we want j, and -1 to account for the increment
                    found = true;
                    break;
                }
            }

            // if we couldn't find it, we're done
            if (!found){
                break;
            }
        }

        // the run is filled, insert it into the right spot of the return data structure
        ret[run[0]->value][run[0]->color] = run;
        index = 0;
    }

    return ret;
}

bool Solve(std::vector<Tile *> startingCombination, std::vector<std::vector<Tile *> >& Board, int range,
std::vector<std::vector<Tile *> > setsMap, std::vector<std::vector<std::vector<Tile *> > > runsSet, std::set<Tile *> visitedTiles);

void removeTileFromContainers(std::vector<Tile *>& startingCombination, std::vector<std::vector<Tile*> >& setsMap, 
std::vector<std::vector<std::vector<Tile *> > >& runsSet, Tile* t, int range) {
    // we remove this tile from the starting combinations, 
    // setsMap and runsSet containers

    auto it = std::find(startingCombination.begin(), startingCombination.end(), t);
    startingCombination.erase(it);

    // remove the tile from its appearances in the runset
    // the runset needs to be readjusted since the data structure will
    // now be invalidated
    std::vector<Tile *>& relevantRun = runsSet[t->value][t->color];
    int max_iteration_count = range;
    int startValue = t->value;
    while (max_iteration_count != 0){
        if (!runsSet[startValue][t->color].empty()){
            break;
        }
        startValue = startValue == 0 ? range-1 : startValue - 1;
        max_iteration_count--;
    }

    // we should be able to find the tile or the tile that owns the run that contains the tile
    assert(max_iteration_count != 0);

    if (runsSet[startValue][t->color].size() == 1){
        runsSet[startValue][t->color].clear();
    }

    else if (startValue == t->value){
        // we need to move the rest of the run to its new location
        std::vector<Tile *> temp(runsSet[startValue][t->color].begin()+1, runsSet[startValue][t->color].end());
        int nextValue = startValue == range-1 ? 0 : startValue + 1;
        runsSet[nextValue][t->color] = temp;
        runsSet[startValue][t->color].clear();
    }
    else {
        // we need to cut off part of the run, and reinsert the rest of the cut-off run
        auto it = std::find(runsSet[startValue][t->color].begin(), runsSet[startValue][t->color].end(), t);
        // we should absolutely find the tile
        assert(it != runsSet[startValue][t->color].end());

        // chop off the rest of the run
        std::vector<Tile *> temp(it+1, runsSet[startValue][t->color].end());
        while (it != runsSet[startValue][t->color].end()) {
            it = runsSet[startValue][t->color].erase(it);    
        }

        // insert the run to its new location
        if (temp.size() != 0){
            runsSet[temp[0]->value][t->color] = temp;
        }
    }

    relevantRun.erase(
        std::remove(relevantRun.begin(), relevantRun.end(), t),
        relevantRun.end());

    // remove the tile from its appearances in the sets map
    std::vector<Tile *>& relevantSet = setsMap[t->value];
    relevantSet.erase(std::remove(relevantSet.begin(), relevantSet.end(), t),
                relevantSet.end());
}

// returns true if t1, which has already been placed on the board, is compatible
// with t2, a tile that is tentatively going to be placed adjacently to t1 on the board, false otherwise
// TODO: Validate that sets contain all different colors. This matters when using 6's as 9's, low priority
bool isValid(Tile *t1, std::vector<std::vector<Tile *> >& Board, int range){
    std::vector<Tile *> v1;
    v1.reserve(3);
    std::vector<Tile *> v2;
    v2.reserve(3);
    std::vector<Tile *> v3;
    v3.reserve(3);

    if (t1->x-2 > 0){
        // ensure there are no empty tiles
        if (Board[t1->x-2][t1->y] && Board[t1->x-1][t1->y] && Board[t1->x][t1->y]){
            v1.push_back(Board[t1->x-2][t1->y]);
            v1.push_back(Board[t1->x-1][t1->y]);
            v1.push_back(Board[t1->x][t1->y]);
        }
    }
    if (t1->x-1 > 0 && t1->x+1 < Board.size()){
        if (Board[t1->x-1][t1->y] && Board[t1->x][t1->y] && Board[t1->x+1][t1->y]){
            v2.push_back(Board[t1->x-1][t1->y]);
            v2.push_back(Board[t1->x][t1->y]);
            v2.push_back(Board[t1->x+1][t1->y]);
        }
    }
    if (t1->x+2 < Board.size()){
        if (Board[t1->x][t1->y] && Board[t1->x+1][t1->y] && Board[t1->x+2][t1->y]){
            v3.push_back(Board[t1->x][t1->y]);
            v3.push_back(Board[t1->x+1][t1->y]);
            v3.push_back(Board[t1->x+2][t1->y]);
        }
    }

    if (!v1.empty() && !v2.empty() && !v3.empty()){
        if ( !(isRun(v1, range) || isSet(v1)) && !(isRun(v3, range) || isSet(v3))
            && !(isRun(v2, range) || isSet(v2)) ){
            return false;
        }
    }

    else if (!v1.empty() && !v2.empty()){
        if(!(isRun(v1, range) || isSet(v1)) || !(isRun(v2, range) || isSet(v2))){
            return false;
        }
    }
    else if (!v2.empty() && !v3.empty()){
        if(!(isRun(v2, range) || isSet(v2)) || !(isRun(v3, range) || isSet(v3))){
            return false;
        }
    }
    else if (!v1.empty()){
        if (!(isRun(v1, range) || isSet(v1))){
            return false;
        }
    }
    else if (!v3.empty()){
        if (!(isRun(v3, range) || isSet(v3))){
            return false;
        }
    }

    std::vector<Tile *> v4;
    v4.reserve(3);
    std::vector<Tile *> v5;
    v5.reserve(3);
    std::vector<Tile *> v6;
    v6.reserve(3);

    // do the y axis
    if (t1->y-2 > 0){
        if (Board[t1->x][t1->y-2] && Board[t1->x][t1->y-1] && Board[t1->x][t1->y]){
            v4.push_back(Board[t1->x][t1->y-2]);
            v4.push_back(Board[t1->x][t1->y-1]);
            v4.push_back(Board[t1->x][t1->y]);
        }
    }
    if (t1->y-1 > 0 && t1->y+1 < Board.size()){
        if (Board[t1->x][t1->y-1] && Board[t1->x][t1->y] && Board[t1->x][t1->y+1]){
            v5.push_back(Board[t1->x][t1->y-1]);
            v5.push_back(Board[t1->x][t1->y]);
            v5.push_back(Board[t1->x][t1->y+1]);
        }
    }
    if (t1->y+2 < Board.size()){
        if (Board[t1->x][t1->y] && Board[t1->x][t1->y+1] && Board[t1->x][t1->y+2]){
            v6.push_back(Board[t1->x][t1->y]);
            v6.push_back(Board[t1->x][t1->y+1]);
            v6.push_back(Board[t1->x][t1->y+2]);
        }
    }

    // in any situation, it is necessary that either 
    // 1. this tile joins two valid groups (adjoined)
    // 2. this tile is the center of a set of 3 tiles, and is part of a valid group

    if (!v4.empty() && !v5.empty() && !v6.empty()){
        if ( !(isRun(v4, range) || isSet(v4)) && !(isRun(v6, range) || isSet(v6))
            && !(isRun(v5, range) || isSet(v5)) ){
            return false;
        }
    }

    else if (!v4.empty() && !v5.empty()){
        if(!(isRun(v4, range) || isSet(v4)) || !(isRun(v5, range) || isSet(v5))){
            return false;
        }
    }
    else if (!v5.empty() && !v6.empty()){
        if(!(isRun(v5, range) || isSet(v5)) || !(isRun(v6, range) || isSet(v6))){
            return false;
        }
    }
    else if (!v4.empty()){
        if (!(isRun(v4, range) || isSet(v4))){
            return false;
        }
    }
    else if (!v6.empty()){
        if (!(isRun(v6, range) || isSet(v6))){
            return false;
        }
    }

    // there should be at least one group of size at least 3
    // in existence, otherwise the tile doesn't belong there.

    if (v1.empty() && v2.empty() && v3.empty() &&
        v4.empty() && v5.empty() && v6.empty()){
        return false;
    }

    return true;
}

// attempt to insert this permutation to the left of the current tile
// then resume solving the board
bool insertAndSolve(std::vector<Tile *> &permutation, std::vector<Tile *> startingCombination,
std::vector<std::vector<Tile *> >& Board, int range, std::vector<std::vector<Tile*> > setsMap,
std::vector<std::vector<std::vector<Tile *> > > runsSet, Tile *curTile, 
std::set<Tile *> visitedTiles, int xDim, int yDim){
    int size = permutation.size();
    if (curTile->x-size >= 0){
        // check the board to make sure there's space for the whole permutation
        bool spaceExists = true;
        for (unsigned int i = 1; i < size+1; i++){
            if (Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)]){
                spaceExists = false;
                break;
            }
        }

        if (spaceExists){
            // tentatively perform the insertion onto the board
            for (unsigned int i = size; i > 0; i--){
                Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)] = permutation[size - i];
                Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)]->x = curTile->x+(i*xDim);
                Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)]->y = curTile->y+(i*yDim);
            }

            // ensure the placed tiles are all valid, even if they collide with other
            // tiles that are part of other groups
            for (unsigned int i = size; i > 0; i--){
                if (!isValid(permutation[size - i], Board, range)){
                    // if invalid, we need to unset the tentatively placed tiles
                    for (unsigned int j = size; j > 0; j--){
                        Board[curTile->x+(j*xDim)][curTile->y+(j*yDim)]->x = -1;
                        Board[curTile->x+(j*xDim)][curTile->y+(j*yDim)]->y = -1;
                        Board[curTile->x+(j*xDim)][curTile->y+(j*yDim)] = nullptr;
                    }
                    return false;
                }
            }

            // after performing the insertion, attempt to solve the rest of the board
            for (unsigned int i = size; i > 0; i--){                
                removeTileFromContainers(startingCombination, setsMap, runsSet, permutation[size - i], range);
            }
            bool result = Solve(startingCombination, Board, range, setsMap, runsSet, visitedTiles);
            // regardless of whether we solve or not, the tentatively placed tiles' locations
            // need to be reset.
            for (unsigned int i = size; i > 0; i--){
                Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)]->x = -1;
                Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)]->y = -1;
                Board[curTile->x+(i*xDim)][curTile->y+(i*yDim)] = nullptr;
            }

            return result;
        }
    }
    return false;
}

bool traverseAndSolveSet(std::vector<Tile *> &permutation, std::vector<Tile *> startingCombination, std::vector<std::vector<Tile *> >& Board, int range,
std::vector<std::vector<Tile*> > setsMap, std::vector<std::vector<std::vector<Tile *> > > runsSet, Tile *curTile, std::set<Tile *> visitedTiles){
    // check above below left and right
    visitedTiles.insert(curTile);
    bool success = false;

    if (curTile->x > 0){
        if (Board[curTile->x-1][curTile->y]
            && visitedTiles.find(Board[curTile->x-1][curTile->y]) == visitedTiles.end()){            
            success = traverseAndSolveSet(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x-1][curTile->y], visitedTiles);
            if (success) 
                return success;
        }
    }
    if (curTile->x < Board.size()-1){
        if (Board[curTile->x+1][curTile->y]
            && visitedTiles.find(Board[curTile->x+1][curTile->y]) == visitedTiles.end()){
            success = traverseAndSolveSet(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x+1][curTile->y], visitedTiles);
            if (success) 
                return success;
        }
    }
    if (curTile->y < Board.size()-1){            
        if (Board[curTile->x][curTile->y+1]
            && visitedTiles.find(Board[curTile->x][curTile->y+1]) == visitedTiles.end()){
            success = traverseAndSolveSet(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x][curTile->y+1], visitedTiles);
            if (success) 
                return success;
        }
    }
    if (curTile->y > 0){
        if (Board[curTile->x][curTile->y-1]
            && visitedTiles.find(Board[curTile->x][curTile->y-1]) == visitedTiles.end()){
            success = traverseAndSolveSet(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x][curTile->y-1], visitedTiles);
            if (success) 
                return success;
        }
    }        
    
    // now try and insert this set directly adjacent to this tile.
    // we will certainly fail if this tile is not the wild tile and it
    // has a differing value than the tile we want to add.
    if (!curTile->isWild() && curTile->value != permutation[0]->value)
        return false;

    // further, we must check that the tile(s) that are placed
    // construct a valid set that is of length at least 3,
    // and that any tiles that are placed that are adjacent to other
    // existing tiles are also valid 
    success |= insertAndSolve(permutation, startingCombination, Board, range,
     setsMap, runsSet, curTile, visitedTiles, -1, 0);
    if (success) 
        return success;
    success |= insertAndSolve(permutation, startingCombination, Board, range, 
    setsMap, runsSet, curTile, visitedTiles, 1, 0);
    if (success) 
        return success;
    success |= insertAndSolve(permutation, startingCombination, Board, range,
     setsMap, runsSet, curTile, visitedTiles, 0, -1);
    if (success) 
        return success;
    success |= insertAndSolve(permutation, startingCombination, Board, range,
     setsMap, runsSet, curTile, visitedTiles, 0, -1);
    
    return success;
}

bool AreConsecutive(int x, int y, int range){
    if (abs(x%range - y%range) == 1){
        return true;
    }
    return false;
}

bool traverseAndSolveRun(std::vector<Tile *> &permutation, std::vector<Tile *> startingCombination, std::vector<std::vector<Tile *> >& Board, int range,
std::vector<std::vector<Tile*> > setsMap, std::vector<std::vector<std::vector<Tile *> > > runsSet, Tile *curTile, std::set<Tile *> visitedTiles){
    // check above below left and right
    visitedTiles.insert(curTile);
    bool success = false;

    if (curTile->x > 0){
        if (Board[curTile->x-1][curTile->y]
            && visitedTiles.find(Board[curTile->x-1][curTile->y]) == visitedTiles.end()){            
            success = traverseAndSolveRun(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x-1][curTile->y], visitedTiles);
            if (success) 
                return success;
        }
    }
    if (curTile->x < Board.size()-1){
        if (Board[curTile->x+1][curTile->y]
            && visitedTiles.find(Board[curTile->x+1][curTile->y]) == visitedTiles.end()){
            success = traverseAndSolveRun(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x+1][curTile->y], visitedTiles);
            if (success) 
                return success;
        }
    }
    if (curTile->y < Board.size()-1){            
        if (Board[curTile->x][curTile->y+1]
            && visitedTiles.find(Board[curTile->x][curTile->y+1]) == visitedTiles.end()){
            success = traverseAndSolveRun(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x][curTile->y+1], visitedTiles);
            if (success) 
                return success;
        }
    }
    if (curTile->y > 0){
        if (Board[curTile->x][curTile->y-1]
            && visitedTiles.find(Board[curTile->x][curTile->y-1]) == visitedTiles.end()){
            success = traverseAndSolveRun(permutation, startingCombination, Board, range, setsMap, runsSet, Board[curTile->x][curTile->y-1], visitedTiles);
            if (success) 
                return success;
        }
    }        
    
    // now try and insert this run directly adjacent to this tile.
    // we will certainly fail if this tile is not the wild tile and it
    // has a differing value than the tile we want to add, or if it 
    // has a differing color.
    if (!curTile->isWild() && (curTile->color != permutation[permutation.size()-1]->color ||
     AreConsecutive(curTile->value, permutation[permutation.size()-1]->value, range))){
        return false;
    }

    // further, we must check that the tile(s) that are placed
    // construct a valid run that is of length at least 3,
    // and that any tiles that are placed that are adjacent to other
    // existing tiles are also valid 
    success |= insertAndSolve(permutation, startingCombination, Board, range,
     setsMap, runsSet, curTile, visitedTiles, -1, 0);
    if (success) 
        return success;
    success |= insertAndSolve(permutation, startingCombination, Board, range, 
    setsMap, runsSet, curTile, visitedTiles, 1, 0);
    if (success) 
        return success;
    success |= insertAndSolve(permutation, startingCombination, Board, range,
     setsMap, runsSet, curTile, visitedTiles, 0, -1);
    if (success) 
        return success;
    success |= insertAndSolve(permutation, startingCombination, Board, range,
     setsMap, runsSet, curTile, visitedTiles, 0, -1);

    return success;
}

std::vector<Tile *> splice(std::vector<Tile *> v, int start, int end){
    std::vector<Tile *> ret(v.begin()+start, v.begin()+end);
    return ret;
}

// assume that before any recursion, the board is in a valid state.
// with this assumption, it is fine to return true when there are
// no remaining tiles in the startingCombination container
bool Solve(std::vector<Tile *> startingCombination, std::vector<std::vector<Tile *> >& Board, int range,
std::vector<std::vector<Tile*> > setsMap, std::vector<std::vector<std::vector<Tile *> > > runsSet, std::set<Tile *> visitedTiles){
    // base case: if you have no tiles remaining to place, your turn is done, 
    // success has been achieved, return true
    if (startingCombination.size() == 0){
        return true;
    }        

    std::vector<std::vector<Tile *> > oldBoard = Board;

    // note: sets map entries aren't sorted largest to smallest
    for (auto v : setsMap){
        // try to place all subsets of each set
        std::vector<std::vector<Tile *> > subsets;
        // we need a permutation of each unique combination
        for (int i = v.size(); i > 0; i--){
            // optimization completed: check that permutations are sorted in largest length order first
            std::vector<std::vector<Tile *> > someCombinations = combo(v, i);
            for (int j = 0; j < someCombinations.size(); j++){
                std::vector<Tile *> combination = someCombinations[j];
                if (combination.size() == 1){
                    subsets.push_back(combination);
                    continue;
                }
                do {
                    // the combination at this point represents a specific permutation
                    subsets.push_back(combination);
                } while (std::next_permutation(combination.begin(), combination.end()));
            }
        }
        for (int i = 0; i < subsets.size(); i++){
            // try to place this particular permutation in order
            // onto the board as a set. 
            // then recurse, and try to solve the whole board
            std::vector<Tile *> permutation = subsets[i];

            // start traversing at the super wild in the center
            Tile *curTile = Board[Board.size()/2][Board.size()/2];
            bool success = traverseAndSolveSet(permutation, startingCombination, Board, range, setsMap, runsSet, curTile, visitedTiles);
            if (success){
                startingCombination.clear();
                return true;
            }
        }
    }

    // undo the changes explored when trying to put down a set first,
    // try to place a run down instead
    Board = oldBoard;

    // place all valid subsequences of the run
    for (int val = 0; val < runsSet.size(); val++){
        for (int color = 0; color < runsSet[val].size(); color++){
            if (runsSet[val][color].empty()){
                continue;
            }
            std::vector<std::vector<Tile *> > allConsecutiveSubsequences;

            // we need all possible subsequences for this specific value and color
            // optimization: consider longest runs first
            for(int subseqSize = runsSet[val][color].size()+1; subseqSize > 0 ; subseqSize--){
                for (int start = 0; start < 1+runsSet[val][color].size()-subseqSize; start++){
                    allConsecutiveSubsequences.push_back(splice(runsSet[val][color], start, start+subseqSize));
                }
            }

            for (int i = 0; i < allConsecutiveSubsequences.size(); i++){
                // try to place this particular permutation in order
                // onto the board as a run. 
                // then recurse, and try to solve the whole board
                std::vector<Tile *> subseq = allConsecutiveSubsequences[i];

                // start traversing at the super wild in the center
                Tile *curTile = Board[Board.size()/2][Board.size()/2];
                bool success = traverseAndSolveRun(subseq, startingCombination, Board, range, setsMap, runsSet, curTile, visitedTiles);
                if (success){
                    startingCombination.clear();
                    return true;
                }
            }
        }

        // implement traverseAndSolveRun
        break;
    }                

    return false;
}

bool getResultFromStartingCombo(std::vector<Tile *> v, std::vector<std::vector<Tile *> >& Board, int range, int colors){
    std::vector<std::vector<Tile*> > setsMap = getSets(v, range);        

    // next perform a run analysis
    std::vector<std::vector<std::vector< Tile *> > > runsSet = getRuns(v, range, colors);    

    std::set<Tile *> visitedTiles;
    
    printTileVec(v);
    return Solve(v, Board, range, setsMap, runsSet, visitedTiles);
}

void testMain(){

    std::vector<Tile *> startingCombination;
    int range = 3;
    int colors = 2;
    Tile *t1 = new Tile(0,1,-1,-1);
    Tile *t2 = new Tile(1,1,-1,-1);
    Tile *t3 = new Tile(2,0,-1,-1);
    Tile *t4 = new Tile(0,1,-1,-1);
    Tile *t5 = new Tile(1,1,-1,-1);
    Tile *t6 = new Tile(2,0,-1,-1);

    std::vector<Tile *> v1;
    std::vector<Tile *> v2;
    v1.push_back(t1);
    v1.push_back(t2);
    v1.push_back(t3);

    v2.push_back(t4);
    v2.push_back(t5);
    v2.push_back(t6);

    int starting_tile_size = v1.size();

    std::cout<<std::endl<<"### RUNNING TEST ###"<<std::endl << std::endl;
    std::vector<std::vector<Tile *> > Board(starting_tile_size*2 + 1, std::vector<Tile *>(starting_tile_size*2 + 1, nullptr));

    if (!Board[starting_tile_size][starting_tile_size]){
        Board[starting_tile_size][starting_tile_size] = new Tile(-1,-1, starting_tile_size, starting_tile_size);
    }
    
    bool result1 = getResultFromStartingCombo(v1, Board, range, colors);

    for (int i = 0; i < 100; i++){
        if (result1 != getResultFromStartingCombo(v2, Board, range, colors)){
            std::cout<<"UNEQUAL OUTCOMES"<<std::endl;
            break;
        }
    }
}

int main(int argc, char* argv[]){
    // if (argc){
    //     testMain();
    //     return 0;
    // }
    std::cout<<"Number of numbers?"<<std::endl;
    std::string range_str = "";
    std::cin >> range_str;
    std::cout<<"Number of colors?"<<std::endl;
    std::string colors_str = "";
    std::cin >> colors_str;
    std::cout<<"Number of starting tiles?"<<std::endl;
    std::string starting_tiles_str = "";
    std::cin >> starting_tiles_str;

    // int num_threads = std::thread::hardware_concurrency();
    // std::cout << "number of threads = " << num_threads << std::endl;
    // std::vector<std::thread> thread_pool;
    // for (int i = 0; i < num_threads; i++)
    // {
    //     thread_pool.push_back(std::thread(&Function_pool::infinite_loop_func, &func_pool));
    // }

    auto start = std::chrono::system_clock::now();
    
    int range = atoi(range_str.data());
    int colors = atoi(colors_str.data());
    int starting_tile_size = atoi(starting_tiles_str.data());
    
    std::cout<<"Range is 0 to " << range - 1 << std::endl;
    std::cout<<"There are " << colors_str << " colors" << std::endl;
    std::cout<<"There are " << starting_tiles_str << " starting tiles " << std::endl;
    
    std::vector<Tile *> allAvailableTiles;
    // create all possible tiles
    for (int i = 0; i < range; i++){
        for (int j = 0; j < colors; j++){
            Tile *t = new Tile(i, j, -1, -1);
            allAvailableTiles.push_back(t);
        }
    }
    
    std::cout<<"Calculating all possible starting configurations"<<std::endl;
    std::vector<std::vector<Tile *> > allStartingTileCombinations = combo(allAvailableTiles, starting_tile_size);
    long long int totalStartingConfigurations = allStartingTileCombinations.size();
    
    auto end = std::chrono::system_clock::now();
 
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
 
    std::cout << "elapsed time for starting configurations: " << elapsed_seconds.count() << "s"
              << std::endl;

    std::cout << totalStartingConfigurations << " starting configurations" << std::endl;


    // for (int i = 0; i < allStartingTileCombinations.size(); i++){
    //     std::cout<< "Combination #" << i << " ";
    //     printTileVec(allStartingTileCombinations[i]);
    // }
    
    std::vector<std::vector<Tile *> > Board(starting_tile_size*2 + 1, std::vector<Tile *>(starting_tile_size*2 + 1, nullptr));

    long long int totalSolveable = 0;
    
    for (long long int i = 0; i < allStartingTileCombinations.size(); i++){
        std::vector<Tile *> startingCombination = allStartingTileCombinations[i];
        if (i%1000 == 0){
            std::cout<< i << std::endl;
        }
        // The grid that's large enough to place all the tiles down
        // on the first round is nxn, where n is the number of starting tiles
        // This approximation may be pushed lower, but we can work with
        // this assumption for now.

        // create a clean board
        // put down the super wild in the middle
        if (!Board[starting_tile_size][starting_tile_size]){
            Board[starting_tile_size][starting_tile_size] = new Tile(-1,-1, starting_tile_size, starting_tile_size);
        }

        
        // perform a greedy analysis and try to
        // create as many runs / sets from the 
        // starting tiles as possible

        // first divide the starting tiles into a map of sets of set tiles.
        std::vector<std::vector<Tile *> > setsMap = getSets(startingCombination, range);        

        // next perform a run analysis
        std::vector<std::vector<std::vector<Tile *> > > runsSet = getRuns(startingCombination, range, colors);

        std::set<Tile *> visitedTiles;
        
        // std::cout << i << ": " << std::endl;
        // printTileVec(startingCombination);


        bool solveable = Solve(startingCombination, Board, range, setsMap, runsSet, visitedTiles);
        if (solveable){
            totalSolveable += 1;
        }
        else{
            //std::cout << "ITERATION " << i << " FAILED" << std::endl;
        }
    }    
    
    double ratio = totalSolveable / (double)totalStartingConfigurations;
    std::cout<<"Percentage that one goes out on the first turn: " << std::endl;
    std::cout<< "(" << totalSolveable << "/" << totalStartingConfigurations << ") = " << 100.0*ratio << "%" << std::endl;        
    
    end = std::chrono::system_clock::now();
 
    elapsed_seconds = end-start;
    end_time = std::chrono::system_clock::to_time_t(end);
 
    std::cout << "Total elapsed time: " << elapsed_seconds.count() << "s"
              << std::endl;

    return 0;
}
