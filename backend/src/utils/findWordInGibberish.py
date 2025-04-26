def find_word_in_gibberish(gibberish, words):
    gibberish = gibberish.strip()

    def lcs_length(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m):
            for j in range(n):
                if s1[i].lower() == s2[j].lower():
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        return dp[m][n]

    for word in words:
        match_letters = lcs_length(gibberish, word)
        match_percentage = (match_letters / len(word)) * 100

        if match_percentage >= 60:
            return word

    return None