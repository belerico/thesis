#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace std;

unordered_map<string, int> get_corpus_frequency(const string &corpus_path)
{
    unordered_map<string, int> freqs{};
    stringstream ss{};
    istream_iterator<string> end;
    ifstream ifs{corpus_path};
    if (ifs)
    {
        for (string line; getline(ifs, line);)
        {
            ss << line;
            for (istream_iterator<string> token(ss); token != end; ++token)
                ++freqs[*token];
            ss.str("");
            ss.clear();
        }
    }
    else
        cerr << "Error reading file: " << corpus_path << endl;
    return move(freqs);
}
int main(int argc, char *argv[])
{
    if (argc == 4)
    {
        int i = 0;
        double c1_lenght, c2_length, fr;
        string word;
        ifstream truth{argv[1]};
        unordered_map<string, int> freqs_c1 = get_corpus_frequency(argv[2]);
        unordered_map<string, int> freqs_c2 = get_corpus_frequency(argv[3]);
        c1_lenght = freqs_c1.size();
        c2_length = freqs_c2.size();
        cout << "There're " << c1_lenght << " unique elements in corpus 1" << endl;
        cout << "There're " << c2_length << " unique elements in corpus 2" << endl;
        for (string line; getline(truth, line);)
        {
            stringstream ss{line};
            while (i < 1 && getline(ss, word, '\t'))
                ++i;
            i = 0;
            if (freqs_c1.count(word) > 0)
                cout << word << ": " << freqs_c1[word] << " times in corpus 1. fr = " << (freqs_c1[word] / c1_lenght) << endl;
            else
                cout << "Error! " << word << " does not appear in corpus 1" << endl;
            if (freqs_c2.count(word) > 0)
                cout << word << ": " << freqs_c2[word] << " times in corpus 2. fr = " << (freqs_c2[word] / c2_length) << endl;
            else
                cout << "Error! " << word << " does not appear in corpus 2" << endl;
        }
    }
    else
    {
        cerr << "USAGE: ./analyze <GROUND_TRUTH_PATH> <CORPUS1_PATH> <CORPUS2_PATH>" << endl;
    }
}