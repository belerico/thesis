#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace std;

map<string, int> get_corpus_frequency(const string &corpus_path)
{
    map<string, int> freqs;
    ifstream ifs{corpus_path};
    if (ifs)
    {
        for (string line; getline(ifs, line, ' ');)
        {
            if (line.length() > 0)
                ++freqs[line];
        }
    }
    else
        cerr << "Error reading file: " << corpus_path << endl;
    return freqs;
}
int main(int argc, char *argv[])
{
    if (argc == 4)
    {
        int i = 0;
        double c1_lenght, c2_length, fr;
        string word;
        ifstream truth{argv[1]};
        map<string, int> freqs_c1 = get_corpus_frequency(argv[2]);
        map<string, int> freqs_c2 = get_corpus_frequency(argv[3]);
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
            try
            {
                fr = freqs_c1.at(word);
                cout << word << ": " << fr << " times in corpus 1. fr = " << (fr / c1_lenght) << endl;
            }
            catch (out_of_range)
            {
                cout << "Error! " << word << " does not appear in corpus 1" << endl;
            }
            try
            {
                fr = freqs_c2.at(word);
                cout << word << ": " << fr << " times in corpus 1. fr = " << (fr / c2_length) << endl;
            }
            catch (out_of_range)
            {
                cout << "Error! " << word << " does not appear in corpus 2" << endl;
            }
        }
    }
    else
    {
        cerr << "USAGE: ./analyze <GROUND_TRUTH_PATH> <CORPUS1_PATH> <CORPUS2_PATH>" << endl;
    }
}