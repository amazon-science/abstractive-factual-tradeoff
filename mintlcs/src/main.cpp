#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

int lcs(const std::vector<std::string> &s1,
	const std::vector<std::string> &s2) {
    int m = s1.size();
    int n = s2.size();
    if (m == 0 || n == 0)
	 return 0;
    std::vector<std::vector<int>> dp(m + 1,
				     std::vector<int>(n + 1));
    int i, j;
    for (i = 0; i <= m; i++) {
	dp[i][0] = 0;
    }
    for (j = 0; j <= n; j++) {
	dp[0][j] = 0;
    }
    for (i = 1; i <= m; i++) {
	for (j = 1; j <= n; j++) {
	    if (s1[i - 1] == s2[j - 1]) {
		dp[i][j] = dp[i - 1][j - 1] + 1;
	    } else {
		if (dp[i - 1][j] >= dp[i][j - 1])
		    dp[i][j] = dp[i - 1][j];
		else
		    dp[i][j] = dp[i][j-1];
	    }
	}
    }
    return dp[m][n];
}

namespace py = pybind11;

PYBIND11_MODULE(mintlcs, m) {
    m.def("lcs", &lcs, R"pbdoc(
        Longest common subsequence
    )pbdoc");
}
