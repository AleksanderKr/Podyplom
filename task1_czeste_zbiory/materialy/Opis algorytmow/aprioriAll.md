# Algorytm AprioriAll

## Cel
Znaleźć częste sekwencje zbiorów elementów, czyli wzorce zakupów w czasie.

## Dane wejściowe
- Zbiór klientów  
- Każdy klient posiada uporządkowaną sekwencję transakcji  
- Każda transakcja to zbiór produktów  
- Kolejność transakcji ma znaczenie  

## Główna idea
1. Najpierw znaleźć częste itemsety (za pomocą Apriori).  
2. Następnie znaleźć częste sekwencje tych itemsetów w czasie.  

Jeżeli sekwencja jest częsta, to wszystkie jej podsekwencje również muszą być częste
(antymonotoniczność dla sekwencji).

---

## Schemat działania

1. Uruchom Apriori → uzyskaj zbiór częstych itemsetów L.
2. Zakoduj każdą transakcję jako zbiór częstych itemsetów, które zawiera.
3. Utwórz częste sekwencje długości 1.
4. Generuj kandydatów długości k z sekwencji długości k−1 (join).
5. Usuń kandydatów, których podsekwencje nie są częste (prune).
6. Policz support: liczba klientów, u których sekwencja występuje w odpowiedniej kolejności.
7. Zakończ, gdy brak nowych sekwencji.

---

## Przykład (min_sup = 2)

### Dane

C1: ⟨ {i1}, {i2} ⟩  
C2: ⟨ {i1,i2}, {i4} ⟩  
C3: ⟨ {i1}, {i2}, {i4} ⟩  

---

## Krok 1 – częste itemsety (Apriori)

Załóżmy, że częste są:

{i1}, {i2}, {i4}, {i1,i2}

---

## Krok 2 – 1-sekwencje

⟨{i1}⟩ → 3  
⟨{i2}⟩ → 3  
⟨{i4}⟩ → 2  

---

## Krok 3 – 2-sekwencje

Sprawdzamy kolejność w czasie:

⟨{i1},{i2}⟩  
C1: tak  
C3: tak  
Support = 2 → częsta

⟨{i2},{i4}⟩  
C3: tak  
Support = 1 → odrzucamy

---

## Wynik

Częsta sekwencja:

⟨{i1},{i2}⟩

---

## Różnica względem Apriori

Apriori znajduje współwystępowanie w jednej transakcji.  
AprioriAll znajduje wzorce kolejności między transakcjami.