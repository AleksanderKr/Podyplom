rnn expl
WYJAŚNIENIE DZIAŁANIA KODU: EKSPLORACJA SEKWENCJI PRZEZ SIECI NEURONOWE

Zaproponowany kod demonstruje, w jaki sposób można zaadaptować rekurencyjną sieć neuronową (LSTM) do rozwiązywania problemów typowych dla klasycznego Data Miningu (algorytmy AprioriAll, SPADE). Rozwiązanie to składa się z trzech kluczowych filarów, które warto omówić ze studentami:

1. Reprezentacja wieloelementowych zbiorów (Itemsets) w czasie:
Tradycyjne sieci RNN przyjmują na wejściu pojedyncze tokeny. W analizie koszykowej/sekwencyjnej, w jednym momencie (pos/time) może wystąpić kilka elementów naraz.
Rozwiązaniem zastosowanym w klasie `SetLSTMModel` jest tzw. Pooling (uśrednianie). Dla każdego "koszyka" sieć pobiera wektory zanurzeń (embeddings) wszystkich należących do niego elementów, a następnie wyciąga z nich średnią arytmetyczną (`embeds.mean(dim=0)`). Dzięki temu, bez względu na to czy pacjent wziął jeden lek, czy trzy leki naraz, krok czasowy reprezentowany jest przez jeden stały wektor, będący "wypadkową" całego zbioru.

2. Przewidywanie całych zbiorów (Multi-label Classification):
Klasyfikacja wieloklasowa zakłada, że odpowiedzią jest tylko jeden, najbardziej prawdopodobny element. Ponieważ chcemy przewidzieć, że po leku A nastąpi zbiór leków {B, C}, musimy umożliwić sieci typowanie wielu odpowiedzi naraz.
Uzyskano to poprzez:
- Zmianę targetu na wektor typu multi-hot (zera wszędzie, jedynki na pozycjach leków występujących w następnym kroku).
- Zastosowanie funkcji błędu `BCEWithLogitsLoss` zamiast `CrossEntropyLoss`.
- Nałożenie funkcji `sigmoid` na wyjście sieci podczas ewaluacji. Każdy unikalny element ze słownika otrzymuje niezależne prawdopodobieństwo z przedziału [0, 1].

3. Ekstrakcja częstych sekwencji (Generowanie Drzewa):
Sieć neuronowa domyślnie służy do przewidywania. Aby działała jak algorytm odkrywający reguły, wprowadzono funkcję `extract_frequent_sequences`.
Działa ona analogicznie do fazy "Join & Prune" w algorytmach z rodziny Apriori:
- Inicjalizuje przeszukiwanie od pojedynczych elementów (1-sekwencje).
- Podaje element do sieci i sprawdza, które kolejne elementy przekraczają zadany próg pewności (threshold). Próg ten stanowi neuronowy odpowiednik parametru `minsup`.
- Elementy spełniające warunek są grupowane w zbiór i dołączane do sekwencji (tworząc 2-sekwencję).
- Proces powtarza się rekurencyjnie (lub w pętli z użyciem kolejki) do osiągnięcia maksymalnej długości (`max_len`), pozwalając sieci "wyśnić" najsilniejsze wzorce ukryte w danych historycznych, zwracając listę częstych sekwencji.