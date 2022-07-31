\begin{table}[htb] \footnotesize
    \begin{center}
        \begin{tabular}{llll}
            \hline
            Method &\quad  Influencer &\quad Speech  &\quad TED talk \\
            \hline
            SCL     &\quad  55.61   &\quad 48.34    &\quad 54.77 \\
            \hline
            LTR     &\quad  70.60   &\quad 51.97   &\quad 58.74 \\
            \hline
            VEC         &\quad  72.37   &\quad 54.01    &\quad 60.12 \\
            \hline
             LSTM         &\quad  72.10   &\quad 51.33    &\quad 59.54 \\
            \hline
            CLSTM-S         &\quad  73.47   &\quad \textbf{61.41}    &\quad \textbf{62.00} \\
            \hline
            CLSTM         &\quad  \textbf{75.71}   &\quad N/A    &\quad N/A \\
            \hline
        \end{tabular}
    \end{center}
    \caption{ROC-AUC comparison between proposed methods and state-of-the-art methods.}
    \label{tab:auc_cmp}
\end{table}