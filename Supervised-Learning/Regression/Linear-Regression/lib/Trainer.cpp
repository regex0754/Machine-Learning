#include <bits/stdc++.h>
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

template <typename T>
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

    bool IsDataValid()
    {
        if (m_trainingData.size() == 0) return true;

        int l_dimensions = std::get<0>(m_trainingData[0]).size();
        for (int i = 1; i < m_trainingData.size(); i++)
        {
            if (std::get<0>(m_trainingData[i]).size() != l_dimensions)
            {
                return false;
            }
        }
        return true;
    }
};

template<typename T>
struct Matrix{
    int r , c;
    std::vector<std::vector<T>> M;

    Matrix(int _n = 0,int _m = 0){
        r = _n;c = _m;
        M.clear();
        M.resize(r , std::vector<T>(c));
    }
    Matrix(const Matrix& rhs){
        r = rhs.r;c = rhs.c;
        M = rhs.M;
    }

    Matrix inline Identity(int x = 0){
        Matrix I(x , x);
        for (int i = 0;i < x;i++)
            I.M[i][i] = 1;
        return I;
    }
    Matrix inline Zero(int x = 0){
        Matrix O(x , x);
        O.set(0);
        return O;
    }

    void inline resize(int _n = 0,int _m = 0){
        r = _n;c = _m;
        M.clear();
        M.resize(r , std::vector<T>(c));   
    }
    void inline random(){
        for (int i = 0;i < r;i++)
            for (int j = 0;j < c;j++)
                M[i][j] = rng();
    }   
    void inline set(const T to = 0){
        for (int i = 0;i < r;i++){
            for (int j = 0;j < c;j++){
                M[i][j] = to;
            }
        }
    }

    Matrix inline add(const Matrix& M1 , const Matrix& M2){
        // assert(M1.r == M2.r);
        // assert(M1.c == M2.c);
        int n = M1.r , m = M1.c;
        Matrix res(n , m);
        for (int i = 0;i < n;i++){
            for (int j = 0;j < m;j++){
                res.M[i][j] = M1.M[i][j] + M2.M[i][j];
            }
        }
        return res;
    }
    Matrix inline sub(const Matrix& M1 , const Matrix& M2){
        // assert(M1.r == M2.r);
        // assert(M1.c == M2.c);
        int n = M1.r , m = M1.c;
        Matrix res(n , m);
        for (int i = 0;i < n;i++){
            for (int j = 0;j < m;j++){
                res.M[i][j] = M1.M[i][j] - M2.M[i][j];
            }
        }
        return res;
    }
    Matrix inline mul(const Matrix& M1 , const Matrix& M2){
        //assert(M1.c == M2.r);
        int nr = M1.r,nc = M2.c , kk = M1.c;
        Matrix res(nr , nc);
        for (int i = 0;i < nr;i++){
            for (int j = 0;j < nc;j++){
                for (int k = 0;k < kk;k++){
                    res.M[i][j] += M1.M[i][k] * M2.M[k][j];
                }
            }
        }
        return res;
    }

    Matrix inline operator + (const Matrix& rhs){
        return add(*this , rhs);
    }
    Matrix inline operator - (const Matrix& rhs){
        return sub(*this , rhs);
    }
    Matrix inline operator * (const Matrix& rhs){
        return mul(*this , rhs);
    }
    Matrix inline operator == (const Matrix& rhs){
        return (M == rhs.M);
    }

    Matrix inline operator * (const T rhs){
        Matrix res(r , c);
        for (int i = 0;i < r;i++){
            for (int j = 0;j < c;j++){
                res.M[i][j] = M[i][j] * rhs;
            }
        }
        return res;
    }
    
    template<typename U>
    Matrix inline pow(U exp){
        if (exp == 1)
            return *this;
        assert(r == c);
        int N = r;
        Matrix res = Identity(N) , x = (*this);
        while(exp > 0){
            if (exp & 1){
                res = (res * x);
            }
            exp >>= 1;
            x = (x * x);
        }
        return res;
    }

    Matrix inline Transpose(const Matrix& rhs){
        assert(rhs.r == rhs.c);
        int n = rhs.r;
        Matrix<T> res(n , n);
        for (int i = 0;i < n;i++){
            for (int j = 0;j < n;j++){
                res.M[i][j] = res.M[j][i];
            }
        }
        return res;
    }

    Matrix Inverse(){
        assert(r == c);
        Matrix A(*this);
        int n = r;
        std::vector<T> col(n);
        std::vector<std::vector<T>> tmp(n, std::vector<T>(n));
        for (int i = 0;i < n;i++){
            tmp[i][i] = 1;col[i] = i;
        }
        for (int i = 0;i < n;i++){
            int r = i, c = i;
            for (int j = i;j < n;j++){
                for (int k = i;k < n;k++){
                    if (A.M[j][k]){
                        r = j; c = k; goto found;       
                    }
                }
            }
            return Zero(r);
    found:
            A.M[i].swap(A.M[r]); tmp[i].swap(tmp[r]);
            for (int j = 0;j < n;j++){
                swap(A.M[j][i], A.M[j][c]);
                swap(tmp[j][i], tmp[j][c]);
            }
            swap(col[i], col[c]);
            T val = 1 / A.M[i][i];
            for (int j = i + 1;j < n;j++){
                T f = A.M[j][i] * val;
                A.M[j][i] = 0;
                for(int k = i + 1;k < n;k++){
                    A.M[j][k] = (A.M[j][k] - f * A.M[i][k]);
                }
                for (int k = 0;k < n;k++){ 
                    tmp[j][k] = (tmp[j][k] - f * tmp[i][k]);
                }
            }
            for (int j = i + 1;j < n;j++){
                A.M[i][j] = A.M[i][j] * val;
            }
            for (int j = 0;j < n;j++){
                tmp[i][j] = tmp[i][j] * val;
            }
            A.M[i][i] = 1;
        }

        for (int i = n - 1; i > 0; --i){
            for (int j = 0;j < i;j++){
                T val = A.M[j][i];
                for (int k = 0;k < n;k++){
                    tmp[j][k] = (tmp[j][k] - val * tmp[i][k]);
                }
            }
        }

        for (int i = 0;i < n;i++){
            for (int j = 0;j < n;j++){
                A.M[col[i]][col[j]] = tmp[i][j];
            }
        }
        //i.e. A.M = tmp;
        return A;
    }
};

template <typename T>
class LinearRegression
{
    // Y = P * X + e | P is the parameter vector, e is the noise
    // Assuming the e has gaussian distribution for this problem and independent
    // Maximizing log likelyhood will give us,
    // P' = P - (aplha / (sample size)) * SUM((Y(P, X(i)) - Y(i)) * Tranpose(X(i))) | Batch gradient descent
    public:
    std::vector<std::tuple<std::vector<T>, T>> m_data;
    std::vector<Matrix<T>> m_trnX;

    LinearRegression() = delete;
    explicit LinearRegression(const std::vector<std::tuple<std::vector<T>, T>> p_trainingData) : m_data(p_trainingData) 
    {
        m_trnX.clear();

        int32_t l_dataSize = static_cast<int32_t>(m_data.size()), n = static_cast<int32_t>(std::get<0>(m_data[0]).size());
        Matrix<T> l_xMat(1, n + 1);
        l_xMat.set(1);

        for (int i = 0; i < l_dataSize; i++)
        {
            std::vector<T>& l_x = std::get<0>(m_data[i]);
            for (int j = 0; j < n; j++)
            {
                l_xMat.M[0][j] = l_x[j];
            }
            m_trnX.push_back(l_xMat);
        }
    }

    T Eval(const Matrix<T>& p_mat1, const Matrix<T>& p_mat2)
    {
        Matrix<T> l_result = p_mat1;
        l_result = l_result * p_mat2;
        return l_result.M[0][0];
    }

    T EvalOptimizationFunction(const Matrix<T>& p_parameter)
    {
        int32_t l_dataSize = static_cast<int32_t>(m_data.size());
        T l_val = 0, diff;
        for (int32_t l_index = 0; l_index < l_dataSize; l_index++)
        {
            diff = (Eval(m_trnX[l_index], p_parameter) - std::get<1>(m_data[l_index]));
            l_val += diff * diff;
        }
        l_val /= l_dataSize;
        return l_val;
    }

    void GradiantDescent(T p_alpha, T p_limit, int p_maxIteration)
    {
        if (m_data.size() == 0)
        {
            return;
        }

        int32_t l_iteration = 0, n = static_cast<int32_t>(std::get<0>(m_data[0]).size()), l_sampleSize = static_cast<int32_t>(m_data.size());

        T Y, l_alpha = p_alpha;
        Matrix<T> l_oldP(n + 1, 1), l_newP(n + 1, 1), l_TrnX(n + 1, 1);

        l_newP.random();
        l_TrnX.set(1);

        do
        {
            std::swap(l_oldP, l_newP);
            l_newP = l_oldP;
            for (int l_index = 0; l_index < l_sampleSize; l_index++)
            {
                l_newP = l_newP - l_TrnX * ((l_alpha / l_sampleSize) * (Eval(m_trnX[l_index], l_oldP) - std::get<1>(m_data[l_index])));
            }
            l_iteration++;
        } while (EvalOptimizationFunction(l_newP) > p_limit && l_iteration < p_maxIteration);
        
        std::cout << std::fixed << std::setprecision(10);
        for (int l_dimension = 0; l_dimension < l_newP.M.size(); l_dimension++)
        {
            if (l_dimension) std::cout << " " << l_newP.M[l_dimension][0];
            else std::cout << l_newP.M[l_dimension][0];
        } std::cout << std::endl;
    }
};

int main()
{
    freopen("../Output", "w", stdout);

    FileReader<double> l_fileReader("../Training_Data.txt");
    l_fileReader.Parse();
    assert(l_fileReader.IsDataValid());

    LinearRegression<double> l_linearRegression(l_fileReader.m_trainingData);
    l_linearRegression.GradiantDescent(0.001, 0.0000000000001, 10000);

    return 0;
}