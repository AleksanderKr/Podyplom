# Algorytm Apriori

## Cel
Znaleźć częste zbiory elementów (itemsety), czyli produkty często kupowane razem w tej samej transakcji.

## Dane wejściowe
- Zbiór transakcji D  
- Każda transakcja to zbiór produktów  
- Kolejność produktów nie ma znaczenia  

## Własność Apriori
Jeżeli zbiór jest częsty, to wszystkie jego podzbiory również muszą być częste.

## Schemat działania
1. Wyznacz częste zbiory 1-elementowe → L1  
2. Wygeneruj kandydatów 2-elementowych → C2 (join)  
3. Usuń kandydatów z nieczęstymi podzbiorami (prune)  
4. Policz wsparcie i utwórz L2  
5. Powtarzaj dla k = 3, 4, …  
6. Zakończ, gdy nie powstają nowe zbiory  

---

## Przykład (min_sup = 2)

### Transakcje
t1: i1 i2 i5  
t2: i2 i4  
t3: i2 i3  
t4: i1 i2 i4  
t5: i1 i3  

---

### Krok 1 – L1
i1 = 3  
i2 = 4  
i3 = 2  
i4 = 2  
i5 = 1 (usuwamy)

L1 = {i1}, {i2}, {i3}, {i4}

---

### Krok 2 – L2
Kandydaci (join):  
{i1,i2}, {i1,i3}, {i1,i4}, {i2,i3}, {i2,i4}, {i3,i4}

Liczymy wsparcie:

{i1,i2} = 2  
{i2,i4} = 2  
pozostałe < 2

L2 = {i1,i2}, {i2,i4}

---

### Krok 3 – L3
Nie można utworzyć zbiorów 3-elementowych spełniających warunek.

Algorytm kończy działanie.

---

## Wynik
Częste zbiory:
- {i1}, {i2}, {i3}, {i4}
- {i1,i2}, {i2,i4}