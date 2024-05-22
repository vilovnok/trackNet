#pragma comment(linker, "/include:?warp_size@cuda@at@@yahxz")

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>

#include <torch/torch.h>
#include <torch/script.h>

#define N_STATIONS 35
#define PRINT true

using namespace std;
using namespace torch::indexing;
using namespace std::chrono;

struct Hit
{
    int hit_id;
    int track;
    int station;
    float x;
    float y;
    float z;
};


vector<vector<Hit>> parse_csv(string path, int n_events) 
{
    // n_events * n_hits * Hit
    vector<vector<Hit>> result;
    
    string delimiter = "\t";
    
    for (int i = 0; i < n_events; i++)
        result.push_back({});
    
    int evt, station, trk;
    float x, y, z, px, py, pz, vtxx, vtxy, vtxz;
    
    ifstream infile(path);
    int hit_id = 0;
    while (infile >> evt >> x >> y >> z >> station >> trk >> px >> py >> pz >> vtxx >> vtxy >> vtxz)
    {
        result[evt].push_back({hit_id, trk, station, x, y, z});
        hit_id++;
        if (PRINT)
        cout << "\r parsing hits: " << hit_id;
    }
    if (PRINT)
    cout << endl;
    return result;
}

vector<vector<Hit>> transform(vector<vector<Hit>> hits_all)
{
    // normalizing hits coords with constraints
    float x_min, x_max, y_min, y_max, z_min, z_max, new_x, new_y, new_z;
    x_min = -851.;
    x_max = 851.;
    y_min = -851.;
    y_max = 851.;
    z_min = -2386.;
    z_max = 2386.;
    
    vector<vector<Hit>> new_hits;
    int i = 0;
    for (auto vect: hits_all)
    {
        new_hits.push_back({});
        for (auto hit: vect)
        {
            new_x = 2 * (hit.x - x_min) / (x_max - x_min) - 1;
            new_y = 2 * (hit.y - y_min) / (y_max - y_min) - 1;
            new_z = 2 * (hit.z - z_min) / (z_max - z_min) - 1;
            new_hits[i].push_back({hit.hit_id, hit.track, hit.station, new_x, new_y, new_z});
            
        }
        i++;
        if (PRINT)
        cout << "\r transforming hits: " << i;
    }
    if (PRINT)
    cout << endl;
    return new_hits;
}

vector<vector<Hit>> combine(vector<vector<Hit>> hits_all, int n_together)
{
    // combining events together by simple concat to model timeslices
    vector<vector<Hit>> result;
    int n_events_new = hits_all.size() / n_together;
    
    for (int i = 0; i < n_events_new; i++)
        result.push_back({});
    
    for (int i = 0; i < hits_all.size(); i++)
    {
        result[i / n_together].insert(result[i / n_together].end(), hits_all[i].begin(), hits_all[i].end());
        if (PRINT)
        cout << "\r combining hits: " << i;
    }
    if (PRINT)
    cout << endl;
    return result;
}

vector<vector<float>> get_hits_by_station(vector<Hit> hits)
{
    // n_stations * (n_hits * 3(xyz) linearized)
    vector<vector<float>> hits_by_station;
    for (int i = 0; i < N_STATIONS; i++)
    {
        hits_by_station.push_back({});
    }
    for (auto hit: hits)
    {
        hits_by_station[hit.station].insert(hits_by_station[hit.station].end(), {hit.x, hit.y, hit.z});
    }
    
    vector<vector<float>> hits_by_station_double;
    for (int i = 0; i < N_STATIONS; i++)
    {
        hits_by_station_double.push_back({});
    }
    for (int i = 0; i < N_STATIONS - 1; i++)
    {
        hits_by_station_double[i].insert(hits_by_station_double[i].end(), hits_by_station[i].begin(), hits_by_station[i].end());
        hits_by_station_double[i].insert(hits_by_station_double[i].end(), hits_by_station[i+1].begin(), hits_by_station[i+1].end());
    }
    hits_by_station_double[N_STATIONS-1].insert(hits_by_station_double[N_STATIONS - 1].end(), hits_by_station[N_STATIONS-1].begin(), hits_by_station[N_STATIONS-1].end());
    return hits_by_station_double;
}

vector<vector<int>> get_indexes_by_station(vector<Hit> hits)
{
    // n_stations * (n_hits * 1(id) linearized)
    vector<vector<int>> indexes_by_station;
    for (int i = 0; i < N_STATIONS; i++)
    {
        indexes_by_station.push_back({});
    }
    for (auto hit: hits)
    {
        indexes_by_station[hit.station].push_back(hit.hit_id);
    }
    
    vector<vector<int>> indexes_by_station_double;
    for (int i = 0; i < N_STATIONS; i++)
    {
        indexes_by_station_double.push_back({});
    }
    for (int i = 0; i < N_STATIONS - 1; i++)
    {
        indexes_by_station_double[i].insert(indexes_by_station_double[i].end(), indexes_by_station[i].begin(), indexes_by_station[i].end());
        indexes_by_station_double[i].insert(indexes_by_station_double[i].end(), indexes_by_station[i+1].begin(), indexes_by_station[i+1].end());
    }
    indexes_by_station_double[N_STATIONS-1].insert(indexes_by_station_double[N_STATIONS - 1].end(), indexes_by_station[N_STATIONS-1].begin(), indexes_by_station[N_STATIONS-1].end());
    
    return indexes_by_station_double;
}

vector<float> get_chunk_data_x(vector<float> first_station_hits)
{
    // n_first_hits * n_stations * 3 linearized
    vector<float> chunk_data_x;
    for (int i = 0; i < first_station_hits.size() / 3; i++)
    {
        auto current_x = first_station_hits.begin() + i*3;
        chunk_data_x.insert(chunk_data_x.end(), current_x, current_x + 3);
        chunk_data_x.insert(chunk_data_x.end(), (N_STATIONS - 1) * 3, 0.);
    }
    return chunk_data_x;
}

vector<int> get_cand_mask(int n_first_station)
{
    // n_first_hits * 1 linearized
    vector<int> cand_mask;
    cand_mask.insert(cand_mask.end(), n_first_station, 1);
    return cand_mask;
}

vector<int> get_hits_indexes_global(vector<int> first_station_indexes)
{
    // n_first_hits * n_stations * 1 linearized
    vector<int> hits_indexes_global;
    for (int i = 0; i < first_station_indexes.size(); i++)
    {
        hits_indexes_global.push_back(first_station_indexes[i]);
        hits_indexes_global.insert(hits_indexes_global.end(), (N_STATIONS - 1), -1);
    }
    return hits_indexes_global;
}


int main(int argc, const char* argv[]) {
    string path = argv[1];//"./output-100k.tsv";
    int n_events = stoi(argv[2]);//100000;
    int combine_together = 40;
    
    vector<vector<Hit>> hits_all = parse_csv(path, n_events);
    hits_all = transform(hits_all);
    hits_all = combine(hits_all, combine_together);
    n_events = n_events / combine_together;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Executing on GPU." << std::endl;
        device = torch::kCUDA;
    }
    else {
        std::cout << "CUDA is not available! Executing on CPU." << std::endl;
    }
    
    torch::jit::script::Module model = torch::jit::load("./tracknet_torchscript_model_cuda.pt", device);
    //model.to(device);
    cout << "model loaded\n";
    model.eval();
    //cout << model.device() << "\n";
    
    vector<double> model_times;
    vector<double> preproc_times;

    for (int event_id = 0; event_id < n_events; event_id++)
    {
        auto ev_start = high_resolution_clock::now();
        // building vectors - separating hits and indexes by station, creating chunk_data_x, hits_indexes_global, cand_mask from first station hits as working vectors
        vector<vector<float>> hits_by_station = get_hits_by_station(hits_all[event_id]);
        vector<vector<int>> indexes_by_station = get_indexes_by_station(hits_all[event_id]);
        vector<float> chunk_data_x = get_chunk_data_x(hits_by_station[0]);
        vector<int> cand_mask = get_cand_mask(indexes_by_station[0].size());
        vector<int> hits_indexes_global = get_hits_indexes_global(indexes_by_station[0]);
        
        
        // transferring vectors to torch tensors, .clone needed because from_blob doesnt take ownership of data
        auto int_options = torch::TensorOptions().dtype(torch::kInt32);
        int n_first_hits = indexes_by_station[0].size();
        
        vector<torch::Tensor> hits_by_station_T;
        for (auto hits: hits_by_station)
        {
            hits_by_station_T.push_back(torch::from_blob(hits.data(), {hits.size() / 3, 3}).clone().to(device));
        }
        vector<torch::Tensor> indexes_by_station_T;
        for (auto indexes: indexes_by_station)
        {
            indexes_by_station_T.push_back(torch::from_blob(indexes.data(), {indexes.size(), 1}, int_options).clone().to(device));
        }
        torch::Tensor chunk_data_x_T = torch::from_blob(chunk_data_x.data(), {n_first_hits, N_STATIONS, 3}).clone().to(device);
        torch::Tensor cand_mask_T = torch::from_blob(cand_mask.data(), {n_first_hits, 1}, int_options).clone().to(device);
        torch::Tensor hits_indexes_global_T = torch::from_blob(hits_indexes_global.data(), {n_first_hits, N_STATIONS}, int_options).clone().to(device);
        
        auto ev_preproc_end = high_resolution_clock::now();
        
        // inferring the model
        auto output = model.forward({chunk_data_x_T, cand_mask_T, hits_indexes_global_T, hits_by_station_T, indexes_by_station_T}).toTensor();
        // output - (N, 35) array of hit indexes for predicted track candidates, -1 if no hit on station (on the last stations for short tracks e. g.)
        //cout << output.index({Slice(0, 10)}) << endl; // prints first 10 cands
        //cout << event_id << " " << output.sizes() << endl;
        
        auto ev_model_end = high_resolution_clock::now();
        
        duration<double, std::milli> preproc_duration = ev_preproc_end - ev_start;
        duration<double, std::milli> model_duration = ev_model_end - ev_preproc_end;
        if (event_id > 5) {
            preproc_times.push_back(preproc_duration.count());
            model_times.push_back(model_duration.count());
        }
    }
    
    double preproc_average = accumulate(preproc_times.begin(), preproc_times.end(), 0.0) / n_events;
    double model_average = accumulate(model_times.begin(), model_times.end(), 0.0) / n_events;
    cout << "Preprocessing time per event: " << preproc_average << " ms, model time per event: " << model_average << " ms" << endl;
    cout << model_times[0] << " " << model_times[1] << " " << model_times[2] << endl;
    return 0;
}