#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
using namespace std;


void readFromFile(vector<int> &home, int &homes, int &max, string name)
{
    ifstream file(name);
    if(!file)
        cerr<< "File open error";
    file >> homes;
    file >> max;
    home.resize(homes);

    int i = 0, value; 
    while (file >> value)
    {
        home[i] = value;
        i++;
    }
    file.close();
}

void printData(vector<int> &home, int homes, int max)
{
    cout<< homes << endl << max << endl;
    for (int i = 0; i < homes; i++)
    {
        cout << home[i] << " "<< endl;
    }
}

void seqPrefixSumFunc(vector<int> &home, int homes, vector<int> &prefixSum)
{
    prefixSum[0] = home[0];
    for(int i = 1; i < homes; i++)
    {
        prefixSum[i] = prefixSum[i - 1] + home[i];
    }
}

void seqFindRange(vector<int> &home, vector<int> &prefixSum, int &startHome, int &endHome, int &candies, int max, int homes)
{   
    vector<vector<int>> table(homes, vector<int>(homes));
    startHome = -1;
    endHome = -1; 
    int diff = numeric_limits<int>::max();

    for(int row = 0; row < homes; row++)
    {
        for(int col = row; col < homes; col++)
        {
            if (row == 0)
                table[row][col] = prefixSum[col];
            else if (row == col)
                table[row][col] = home[col];
            else 
                table[row][col] = table[0][col] - table[0][row - 1]; 

            if (table[row][col] <= max && diff > max - table[row][col])
            {
                diff = max - table[row][col];
                startHome = row + 1; 
                endHome = col + 1; 
                candies = table[row][col];

            }
            // if (diff == 0)
            //     break;

        }
    }
}

void parPrefixSumFunc(vector<int> &home, int homes, vector<int> &prefixSum)
{   
    int treeHeight = (int)ceil(log2(homes));

    #pragma omp parallel for
    for (int i = 0; i < homes; i++)
    {
        prefixSum[i] = home[i];
    }

    vector<int> hold;
    hold.resize(homes);

    for (int h = 1; h <= treeHeight; h++)
    {
        int stepSize = 1 << (h - 1);

        
        #pragma omp parallel for
        for (int i = stepSize; i < homes; i++)
        {
            hold[i] = prefixSum[i] + prefixSum[i - stepSize];
        }
        #pragma omp parallel for
        for (int i = stepSize; i < homes; i++)
        {
            prefixSum[i] = hold[i];
        }
    }
}

struct rangeGrabber{
    int candies;
    int start;
    int end;
    rangeGrabber() : candies(-1), start(-1), end(-1){}
};

#pragma omp declare reduction(closestMax : rangeGrabber : omp_out = omp_in.candies > omp_out.candies? omp_in : omp_out)


void parFindRange(vector<int> &home, vector<int> &prefixSum, int &startHome, int &endHome, int &candies, int max, int homes)
{

    vector<vector<int>> table(homes, vector<int>(homes));
    startHome = -1;
    endHome = -1; 
    int diff = numeric_limits<int>::max();

    #pragma omp parallel
    for(int col = 0; col < homes; col++)
    {
        table[0][col] = prefixSum[col];
    }
    
    rangeGrabber grabber;
    #pragma omp parallel for reduction(closestMax:grabber)
    for(int row = 0; row < homes; row++)
    {
        for(int col = row; col < homes; col++)
        {
            if (row != 0)
                table[row][col] = prefixSum[col] - prefixSum[row - 1]; 

            if (table[row][col] > grabber.candies && table[row][col] <= max)
            {
                grabber.candies = table[row][col];
                grabber.start = row + 1;
                grabber.end = col + 1;
            }
        }
    }

    startHome = grabber.start;
    endHome = grabber.end;
    candies = grabber.candies;

}

void printOutput(int startHome, int endHome, int candies)
{
    if (startHome == -1 && endHome == -1)
        cout<< "Don\'t go here"<< endl;
    else
        cout<< "Start at home " << startHome << " and go to home " << endHome << " getting " << candies <<" pieces of candy" << endl;
}

int serialCode(vector<int> &home, vector<int> &prefixSum, int &startHome, int &endHome, int &candies, int max, int homes)
{
    seqPrefixSumFunc(home, homes, prefixSum);
    auto start = std::chrono::high_resolution_clock::now();
    seqFindRange(home, prefixSum, startHome, endHome, candies, max, homes);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printOutput(startHome, endHome, candies);
    return duration.count();
}

int parallelCode(vector<int> &home, vector<int> &prefixSum, int &startHome, int &endHome, int &candies, int max, int homes)
{
    parPrefixSumFunc(home, homes, prefixSum);
    auto start = std::chrono::high_resolution_clock::now();
    parFindRange(home, prefixSum, startHome, endHome, candies, max, homes);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printOutput(startHome, endHome, candies);
    return duration.count();
}



int main()
{
    static vector<int> home;
    static int homes;
    static int max;
    string fileName = "input.txt";
    readFromFile(home, homes, max, fileName);
    vector<int> prefixSum(homes);
    int startHome, endHome, candies;

    int timeSeq = serialCode(home, prefixSum, startHome, endHome, candies, max, homes);

    int timePar = parallelCode(home, prefixSum, startHome, endHome, candies, max, homes);

    cout<< "Serial: "<< timeSeq << endl <<"Parallel: "<<timePar << endl;
    return 0; 
}