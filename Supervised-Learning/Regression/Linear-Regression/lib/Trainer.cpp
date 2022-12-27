#include <bits/stdc++.h>

template <typename T = int>
class FileReader
{
    public:
    std::string m_fileName;
    std::vector<std::tuple<std::vector<T>, T>> m_trainingData;

    FileReader() = delete;
    explicit FileReader(const std::string p_fileName) : m_fileName(p_fileName) {};

    T converter(const std::wstring& p_token)
    {
        return stoi(p_token);
    }

    std::tuple<std::vector<T>, T> GetTuple(const std::wstring& p_line)
    {
        std::vector<T> l_variableX;
        T l_variableY;
        std::wstring l_token = L"";

        for (const wchar_t& l_character : p_line)
        {
            if (l_character == L' ')
            {
                l_variableX.push_back(converter(l_token));

                l_token = L"";
                continue;
            }
            l_token += l_character;
        }
        assert(l_token != L"");
        l_variableY = converter(l_token);

        return std::make_tuple(l_variableX, l_variableY);
    }

    void Parse()
    {
        std::vector<std::tuple<std::vector<T>, T>>().swap(m_trainingData);

        std::wstring l_line;
        std::wifstream l_stream;

        l_stream.open(m_fileName);

        while (getline(l_stream, l_line))
        {
            m_trainingData.push_back(GetTuple(l_line));
        }

        l_stream.close();
    }
};

int main()
{
    freopen("../Output", "w", stdout);

    FileReader<int> l_fileReader("../Training_Data.txt");
    l_fileReader.Parse();

    return 0;
}