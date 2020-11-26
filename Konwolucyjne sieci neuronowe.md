# Głębokie uczenie maszynowe i sieci konwolucyjne - znajdowanie na obrazie konkretnego obiektu


W niniejszym sprawdozdaniu zostanie przedstawiony sposób działania głębokiego uczenia maszynowego w połączeniu z sieciami konwolucyjnymi. Przyjrzyżmy się po krótce całemu obszarowi sztucznej inteligencji, uczenia maszynowego, stosowanych technik i podejść oraz znanych zastosowań i ograniczeń. Przy użyciu dostępnych dzisiaj narzędzi "nakarmimy" i zaimplementujemy w prosty sposób program rozpoznający obiekty (ptaki w naszym przypadku) na obrazach przy użyciu wspomnianych technik. Wyjaśnimy, skrótowo i po części, jak aplikacja np. taka jak Google Photos umożliwia wyszukiwanie zdjęć na podstawie tego, co jest na obrazie:


|  Google umożliwia teraz wyszukiwanie własnych zdjęć według opisu - nawet jeśli nie są one otagowane  |
|---|
|![](https://lh3.googleusercontent.com/3P13tIcHi2IeTAYnErANOo1Hz9O2avwQ2KqeT8y2uyEnuqLWC_6L-hj-vu_DXnEAfUepy66fPOTddsugW9APGbpvHhrBhuqxCeldBClT72WGWt3ibCAzwbYOo8kPTmvjngu5J5OAizBfozPiQFr_XtiJKiRLO45hoaPAZ33D4PwEvyLC77MkdWOZD9hU07TMcNXLlrJw26kvxrcQGUEArg3UUhuPjbqu3GWkQnCGHgLcE53C2ga2tCXLZd3yDcGVoN93tR_hIb5WeAEgadpaK-c_6lnN_KSxT_ijsNfe_CtSS4Y5VcdMjjZruAp4HUpD5uFo-d0R2FfDGRQuj0-g3AAXFZBH1BR7nTYXT1xVkM76wsiSEUSNklMLh-oJKN0jFdXtDmFbemkaEFabYDNGjah9ErmhvWE6tvtXXMYnGwlEaVpxT8ql0dRFcASE9XFib62zaY5QgI2-b6lVM-PyXkrUokc9ucX8VilBJif3exO_drfN2IIjoMqYN9oRvzGkqOZcuxiD4gZ_Z5ysIBs-KGhRpJBfx5FsiegsE6g-ZR8EB6rHpPA6o41I5fm_0aNvdVY9EPWJsfxcgSdSTM3SUqUcOf_vGGOE2c3ihW8isfKhF6yVf7PPmWx9Gold-tix7u0DX7F8R08pnBqDtLvuRRcIMuuK7SJ5eEyHRMA0vlxbWfxgKNBtxAINhFu8=w1606-h432-no?authuser=0)|



## Uczenie maszynowe
W skrócie, polega na tym, że istnieją ogólne algorytmy, które mogą powiedzieć coś interesującego o zestawie danych bez konieczności pisania niestandardowego kodu specyficznego dla problemu. Zamiast pisać kod, podajemy dane do ogólnego algorytmu, który buduje własną logikę na podstawie danych.

Na przykład jednym z rodzajów takich algorytmów jest algorytm klasyfikacji. Może umieszczać dane w różnych grupach. Ten sam algorytm klasyfikacji używany do rozpoznawania odręcznie napisanych numerów może być również używany do klasyfikowania wiadomości e-mail jako spam i nie-spam bez zmiany wiersza kodu. Jest to ten sam algorytm, ale przyjmuje różne dane szkoleniowe, więc ma inną logikę klasyfikacji.



**Uczenie maszynowe** jest to jednak bardzo duży obszar, z zakresu sztucznej inteligencji, obejmujący wiele rodzajów podejść. Np. *Uczenie nadzorowane* i *Uczenie bez nadzoru* czy *Uczenie się ze wzmocnieniem* (oraz inne). Wiele alorytmów które poprawiają się automatycznie poprzez doświadczenie, budowanie modelów matematycznych na podstawie przykładowych danych w celu prognozowania lub podejmowania decyzji bez bycia bezpośrednio przez człowieka zaprogramowanym do tego.

Na przestrzeni lat różni pionierzy z tej dziedziny definiowali pojęcie ML na różne sposoby  chociażby:
> **Herbert Simon** (1983)
Uczenie się oznacza zmiany w systemie, które mają charakter adaptacyjny w tym sensie, że pozwalają systemowi wykonać za następnym razem takie samo zadanie lub zadania podobne bardziej efektywnie.
>

> **Ryszard Michalski** (1986)
"Uczenie się to konstruowanie i zmiana reprezentacji doświadczanych faktów. W ocenie konstruowania reprezentacji bierze się pod uwagę: wiarygodność – określa stopień w jakim reprezentacja odpowiada rzeczywistości, efektywność – charakteryzuje przydatność reprezentacji do osiągania danego celu, poziom abstrakcji – odpowiada zakresowi szczegółowości i precyzji pojęć używanych w reprezentacji; określa on tzw. moc opisową reprezentacji. Reprezentacja jest rozumiana jako np. opisy symboliczne, algorytmy, modele symulacyjne, plany, obrazy."
>
> **Donald Michie** (1991)
System uczący się wykorzystuje zewnętrzne dane empiryczne w celu tworzenia i aktualizacji podstaw dla udoskonalonego działania na podobnych danych w przyszłości oraz wyrażania tych podstaw w zrozumiałej i symbolicznej postaci.
>

*Od 2020 r. Deep learing stało się dominującym podejściem w wielu bieżących pracach w dziedzinie ML. Od tego czasu Wiele źródeł w omawianej dziedzinie nadal twierdzi, że ML pozostaje poddziedziną AI. Główna różnica zdań dotyczy tego, czy całe ML jest częścią AI. Inni uważają, że nie wszystkie obszary ML są częścią AI, gdzie tylko „inteligentny” podzbiór ML jest częścią AI*



| ML jako poddziedzina AI   | Część ML jako poddziedzina AI lub część AI jako poddziedzina ML|
|:-------------------------:|:-------------------------:|
 | ![](https://upload.wikimedia.org/wikipedia/commons/f/fe/Fig-X_All_ML_as_a_subfield_of_AI.jpg) | ![](https://upload.wikimedia.org/wikipedia/commons/2/23/Fig-y_Part_of_ML_as_subfield_of_AI_or_AI_as_subfield_of_ML.jpg) |

Ponieważ uczenie maszynowe staje się coraz ważniejsze w coraz większej liczbie branż, różnicą między dobrą a słabą aplikacją będzie ilość danych potrzebnych do trenowania modeli. Dlatego firmy takie jak Google czy Facebook oferują wiele "darmowych" usług, które zapewniają zwykłemu użytkownikowi wystarczającą ilość miejsca na przechowywanie danych takich jak np. zdjęcia.

## Uczenie głębokie 
Uczenie głębokie jest częścią szerszej rodziny metod uczenia maszynowego opartych na sztucznych sieciach neuronowych z uczeniem reprezentacyjnym. Przymiotnik „głęboki” w głębokim uczeniu się odnosi się do użycia wielu warstw w sieci neuronowej. Głęboka sieć neuronowa (DNN) to sztuczna sieć neuronowa (ANN) z wieloma warstwami między warstwą wejściową i wyjściową. Istnieją różne typy sieci neuronowych, ale zawsze składają się one z tych samych komponentów: neuronów, synaps, wag, "bias" i funkcji. Architektury uczenia głębokiego, takie jak głębokie sieci neuronowe, rekurencyjne sieci neuronowe i konwolucyjne sieci neuronowe, znalazły zastosowanie w takich dziedzinach, jak rozpoznawanie obrazów, systemy wizyjne, rozpoznawanie mowy, przetwarzanie języka naturalnego, rozpoznawanie dźwięku, filtrowanie sieci społecznościowych i wiele innych.

## Sieci neuronowe
Struktury matematyczne i ich programowe lub sprzętowych modele, realizujące obliczenia lub przetwarzanie sygnałów poprzez rzędy elementów przetwarzających, zwanych sztucznymi neuronami, wykonujących pewną podstawową operację na swoim wejściu. Oryginalną inspiracją takiej struktury była budowa naturalnych neuronów, łączących je synaps, oraz układów nerwowych, w szczególności mózgu.
| Uproszczony schemat jednokierunkowej sieci neuronowej. Poszczególne „kółka” oznaczają sztuczne neurony.  | 
|:-------------------------:|
 |![](https://upload.wikimedia.org/wikipedia/commons/3/3c/Neuralnetwork.png) |
 
## Konwolucyjna sieć neuronowa
 
| Typowa architektura CNN  | 
|:---:|
|![](https://miro.medium.com/max/1250/1*vkQ0hXDaQv57sALXAJquxA.jpeg) |
 
 
 Konwolucyjna sieć neuronowa (ConvNet / CNN) to algorytm głębokiego uczenia się, który może pobierać obraz wejściowy, przypisywać znaczenie (możliwe do nauczenia się wagi i odchylenia) różnym aspektom / obiektom obrazu i być w stanie odróżnić jeden od drugiego. Przetwarzanie wstępne wymagane w ConvNet jest znacznie niższe w porównaniu z innymi algorytmami klasyfikacji. Podczas gdy w prymitywnych metodach filtry są tworzone ręcznie, po odpowiednim przeszkoleniu CNN mają zdolność uczenia się tych filtrów / charakterystyk.

Architektura CNN jest analogiczna do struktury połączeń neuronów w ludzkim mózgu i została zainspirowana organizacją rdzenia wzrokowego. Poszczególne neurony reagują na bodźce tylko w ograniczonym obszarze pola widzenia, znanym jako pole odbioru. Zbiór takich pól nakłada się na cały obszar widzenia.


### 1. Obraz wejściowy
Obraz to nic innego jak macierz wartości pikseli, hmm? Dlaczego więc nie spłaszczyć obrazu (np. Matrycy 3x3 do wektora 9x1) i nie przesłać go do perceptronu wielopoziomowego w celu klasyfikacji? Uh .. nie bardzo.


 | Spłaszczenie macierzy obrazu 3x3 do wektora 9x1 | 
|:---:|
 |![](https://miro.medium.com/max/531/1*GLQjM9k0gZ14nYF0XmkRWQ.png) |


W przypadku skrajnie podstawowych obrazów binarnych metoda może wykazywać średnią precyzję. Ale miałaby niewielką lub żadną dokładność, jeśli chodzi o złożone obrazy z zależnościami pomiędzy pikselami.

CNN jest w stanie **skutecznie uchwycić zależności przestrzenne i czasowe** na obrazie poprzez zastosowanie odpowiednich filtrów. Architektura takiej sieci zapewnia lepsze dopasowanie do zbioru danych obrazu ze względu na zmniejszenie liczby zaangażowanych parametrów i możliwość ponownego wykorzystania wag. Innymi słowy, sieć można wyszkolić, aby lepiej rozumiała złożoność obrazu.

| Mamy obraz RGB, który został oddzielony trzema płaszczyznami kolorów - czerwoną, zieloną i niebieską. Istnieje wiele takich przestrzeni kolorów, w których istnieją obrazy - skala szarości, RGB, HSV, CMYK itp.  | 
|:---:|
|![](https://miro.medium.com/max/625/1*15yDvGKV47a0nkf5qLKOOQ.png) |

Można sobie wyobrazić, jak bardzo są to rzeczy wymagające dużej mocy obliczeniowej, gdy obrazy osiągną wymiary, powiedzmy 8K (7680 × 4320). Rolą CNN jest zredukowanie obrazów do postaci, która jest łatwiejsza do przetworzenia, bez utraty informacji, które są kluczowe dla uzyskania dobrej prognozy. Jest to ważne, gdy mamy zaprojektować architekturę, która jest skalowalna do ogromnych zbiorów danych.


### 2. Warstwa konwolucyjna, The Kernel (jądro)

Wymiary obrazu = 5 (wysokość) x 5 (szerokość) x 1 (liczba kanałów, np. RGB)

| Przekształcanie obrazu 5x5x1 z jądrem 3x3x1 w celu uzyskania funkcji konwolucyjnej 3x3x1| 
|:---:|
|![](https://miro.medium.com/max/625/1*GcI7G-JLAQiEoCON7xFbhg.gif) |

W powyższej animacji sekcja zielona odwzorowywuje nasz **obraz wejściowy 5x5x1.** Element odpowiedzialny za wykonanie operacji mnożenia nazywany jest **jądrem / filtrem, K**, reprezentowany kolorem żółtym. W tym przypadku **K to macierz 3x3x1.**

```
Kernel/Filter, K = 
1  0  1
0  1  0
1  0  1
```

Jądro przesuwa się 9 razy z powodu **długości kroku = 1 (bez kroku)**, za każdym razem wykonując operację **mnożenia macierzy między K a częścią obrazu, nad którym jądro się przesuwa (0 lub 1)**. Wynikiem operacji jest suma wykonanych mnożeń (w powyższym przypadku dla każdego max = 5, min = 0).
Filtr przesuwa się w prawo z określoną wartością kroku, aż przeanalizuje całą szerokość. Przechodząc dalej, przeskakuje w dół do początku (do lewej) obrazu z tą samą wartością kroku i powtarza ten proces aż do przejścia całego obrazu.



|Operacja konwolucji na macierzy obrazu MxNx3 z jądrem 3x3x3| 
|:---:|
|![](https://miro.medium.com/max/875/1*ciDgQEjViWLnCbmX-EeSrA.gif) |

Celem operacji konwolucji jest wyodrębnienie cech wysokiego poziomu, takich jak krawędzie, z obrazu wejściowego. Sieci konwolucyjne nie muszą być ograniczone tylko do jednej warstwy. Tradycyjnie, pierwsza warstwa jest odpowiedzialna za przechwytywanie cech niskiego poziomu, takich jak krawędzie, kolor, orientacja gradientu itp. Dzięki dodawaniu warstw architektura dostosowuje się również do funkcji wysokiego poziomu, dając nam sieć, która ma pełnowartościowe zrozumienie obrazów w zbiorze danych.

|Operacja konwolucji z długością kroku = 2| 
|:---:|
|![](https://miro.medium.com/max/494/1*1VJDP6qDY9-ExTuQVEOlVg.gif) |



### 3. Pooling Layer

Podobnie jak warstwa konwolucyjna, Pooling layer jest odpowiedzialna za zmniejszenie rozmiaru przestrzennego elementu konwolucyjnego. Ma to na celu zmniejszenie mocy obliczeniowej wymaganej do przetwarzania danych poprzez redukcję wymiarowości. Ponadto jest przydatny do wyodrębniania dominujących cech, które są niezmienne rotacyjnie i pozycyjnie, a tym samym podtrzymuje proces efektywnego uczenia modelu.


|Pooling 3x3 na funkcji splotu 5x5| 
|:---:|
|![](https://miro.medium.com/max/495/1*uoWYsCV5vBU8SHFPAPao-w.gif) |

Istnieją dwa rodzaje operacji Polling'u: Max pooling i Average Pooling. Max Pooling zwraca maksymalną wartość z części obrazu "pokrytej" przez jądro (kernel). Z drugiej strony, Average Pooling zwraca średnią wszystkich wartości z części obrazu "pokrytej" przez jądro.


# Rozpoznawanie obrazów
### Na początek proste rozpoznawanie pisma odręcznego (liczby osiem)
W ML do uczenia sieci neuronowych zazwyczaj używamy liczb. Ale teraz chcemy przetwarzać obrazy przy użyciu tychże sieci. Jak wprowadzić obrazy do sieci neuronowej zamiast liczb? Sieć neuronowa przyjmuje liczby jako dane wejściowe. Dla komputera obraz jest w rzeczywistości tylko siatką liczb, które reprezentują, jak ciemny jest każdy piksel

 | Aby przesłać obraz do naszej sieci neuronowej, po prostu traktujemy obraz 18x18 pikseli jako tablicę 324 liczb  | 
|:---:|
 |![](https://miro.medium.com/max/1250/1*UDgDe_-GMs4QQbT8UopoGA.png) |
 


|| 
|:---:|
|![](https://miro.medium.com/max/875/1*b31hqXiBUjIXo2HSn_grFw.png) |
| Aby przeprocesować 324 węzłów, po prostu powiększymy naszą sieć neuronową, aby miała 324 węzły wejściowe | 

Nasza sieć neuronowa ma teraz również dwa wyjścia (zamiast tylko jednego). Pierwszy wynik będzie przewidywał prawdopodobieństwo, że obraz jest „8”, a drugi wynik będzie przewidywał prawdopodobieństwo, że obraz nie jest „8”. Mając osobne dane wyjściowe dla każdego typu obiektu, który chcemy rozpoznać, możemy użyć sieci neuronowej do klasyfikacji obiektów w grupy.
Każdy nowoczesny komputer może obsłużyć taką sieć neuronową z kilkuset węzłami w ułamek sekundy.
Teraz pozostaje tylko wytrenować sieć neuronową za pomocą obrazów „8”, i tych, które nie są „8”, aby nauczyła się je rozróżniać. Kiedy wprowadzimy „8”, "powiemy" naszej sieci, że prawdopodobieństwo, że obraz jest „8” to 100%, a prawdopodobieństwo, że nie jest to „8”, wynosi 0%.

 | Tak mógłby wyglądć przykładowy zestaw danych treningowych  | 
|:---:|
 |![](https://miro.medium.com/max/1250/1*vEVQDKp9MgZkVPK4M70EhA.jpeg) |

Taką sieć neuronową możemy wytrenować w kilka minut na nowoczesnym laptopie. Po zakończeniu będziemy mieć sieć neuronową, która potrafi rozpoznawać obrazy przedstawiające „8” z całkiem dużą dokładnością.

#### Tunnel Vision
Zwykłe podanie pikseli do sieci neuronowej pomogło zbudować rozpoznawanie obrazu. Chociaż, to nie jest takie proste.

 | Dobrą wiadomością jest to, że nasza sieć rozpoznawania „8” naprawdę działa dobrze na prostych obrazach, w których litera znajduje się dokładnie na środku obrazu | 
|:---:|
 |![](https://miro.medium.com/max/875/1*5ciREAL7xdyXcD-cSRP7Jw.png) |
 
 
Zła wiadomość: nasza sieć rozpoznawania „8” całkowicie nie działa, gdy litera nie jest idealnie wyśrodkowana na obrazie


 |  Nawet najmniejsza zmiana pozycji psuje wszystko | 
|:---:|
 |![](https://miro.medium.com/max/875/1*b5jMTAiyVhOIB9hheXhMmA.png) |
 
 Dzieje się tak, ponieważ nasza sieć nauczyła się tylko wzoru doskonale wyśrodkowanej „ósemki”. Nie ma absolutnie żadnego pojęcia, czym jest poza centrum „ósemka”. Zna dokładnie jeden wzór i tylko jeden wzór. To nie jest zbyt przydatne. Musimy więc dowiedzieć się, jak sprawić, by nasza sieć neuronowa działała w przypadkach, gdy „8” nie jest idealnie wyśrodkowana.
 
##### Pomysł 1: Wyszukiwanie za pomocą "Sliding Window"
Możemy po prostu przeskanować cały obraz w poszukiwaniu możliwych „8” w mniejszych sekcjach, po jednej sekcji na raz, aż znajdziemy jedną. To rozwiązanie typu "brute force". Działa dobrze w niektórych ograniczonych przypadkach, ale jest naprawdę nieefektywny. Trzeba nieustannie sprawdzać ten sam obraz, szukając obiektów o różnych rozmiarach. Można zrobić to lepiej.

##### Pomysł  2: Więcej danych i głęboka sieć neuronowa
Kiedy trenowaliśmy naszą sieć, pokazaliśmy tylko „8”, które były idealnie wyśrodkowane. Co jeśli wytrenujemy go z większą ilością danych, w tym z „8” we wszystkich różnych pozycjach i rozmiarach na całym obrazie? Nie trzeba nawet zbierać nowych danych treningowych. Można po prostu napisać skrypt, który wygeneruje nowe obrazy z „8” we wszystkich rodzajach różnych pozycji obrazu

 |  Stworzyliśmy syntetyczne dane treningowe, tworząc różne wersje obrazów szkoleniowych, które już mieliśmy | 
|:---:|
 |![](https://miro.medium.com/max/875/1*biD9eS5eB6zXzieonNk-VQ.png) |
 
 
Korzystając z tej techniki, można łatwo stworzyć nieskończoną ilość danych szkoleniowych.
Więcej danych niestety utrudnia rozwiązywanie problemu przez sieć neuronową, ale możemy to zrekompensować, zwiększając naszą sieć, a tym samym zwiększając jej zdolność do uczenia się bardziej skomplikowanych wzorców.


 | Aby powiększyć sieć, wystarczy dodawać kolejne warstwy węzłów | 
|:---:|
 |![](https://miro.medium.com/max/875/1*wfmpsoFqWKC7VadjTJxwnQ.png) |
 
 W ten sposób dochodzimy do utworzenie głebokiej sieci neuronowej. Wraz z zastosowaniem kart graficznych uczenie tak skomplikowanych sieci stało o wiele szybsze. Mimo to istnieje rozwiązanie, które pozwala w inteligentny sposób ominąć ten problem i znacznie ułatwić działanie tego typu sieci.
 
 ## Konwolucja
 
 >Dla wygody (dostępność materiałów zaczerpniętych z artykułu) pojęcie konwolucyjnych sieci neuronowych, w tym rozdziale, zostanie przedstawione na przykładzie zdjęcia dziecka. 
 >
 Zmienimy naszą sieć tak by poprawnie klasyfikowała obraz. Oraz pokażemy jak badać jej skuteczność. Musimy dać naszej sieci neuronowej zrozumienie niezmienności translacji - „8” to „8” (lub inny szukany obiekt) bez względu na to, gdzie na obrazku się pojawia. Zrobimy to za pomocą procesu zwanego Konwolucją

### Jak działa konwolucja
Zamiast dostarczać całe obrazy do naszej sieci neuronowej jako jedną siatkę liczb, wykorzystamy fakt, że obiekt jest taki sam bez względu na to, gdzie się pojawia na obrazie.

##### Krok 1: Dzielimy obraz na nakładające się kafelki
Podobnie jak w przypadku naszego wyszukiwania "Sliding window" powyżej, przesuńmy okno nad całym oryginalnym obrazem i zapiszmy każdy wynik jako oddzielny, mały kafelek obrazu:


 | W ten sposób zamieniliśmy nasz oryginalny obraz w 77 małych kafelków z obrazkami o jednakowych rozmiarach | 
|:---:|
 |![](https://miro.medium.com/max/875/1*xS7EugfgQHk68iph7GHpQg.png) |


##### Krok 2: Wprowadzamy każdy kafelek obrazu do małej sieci neuronowej

Wcześniej podawaliśmy pojedynczy obraz do sieci neuronowej, aby sprawdzić, czy jest to „8”. Zrobimy dokładnie to samo, ale zrobimy to dla każdego kafelka obrazu:

 |Powtarzamy to 77 razy, raz dla każdej płytki | 
|:---:|
 |![](https://miro.medium.com/max/875/1*84-TdHvtAHkfnzwa1ZsTVg.png) |
 
 Jest jednak jedna duża zmiana: zachowamy te same wagi sieci neuronowej dla każdego kafelka na tym samym oryginalnym obrazie. Innymi słowy, każdy kafelek traktujemy jednakowo. Jeśli na jakimś kafelku pojawi się coś interesującego, oznaczymy go jako interesujący.


##### Krok 3: Zapisz wyniki z każdego kafelka w nowej tablicy

Nie chcemy stracić orientacji w ułożeniu oryginalnych płytek. Tak więc zapisujemy wynik przetwarzania każdego kafelka w siatkę w tym samym układzie, co oryginalny obraz

 |Wygląda to tak | 
|:---:|
 |![](https://miro.medium.com/max/875/1*tpMqyjAFgsYWpvlNkZgFfw.png) |
 
 Innymi słowy, zaczęliśmy od dużego obrazu, a skończyliśmy na nieco mniejszej tablicy, która rejestruje, które sekcje naszego oryginalnego obrazu były najbardziej interesujące.

##### Krok 4: Downsampling

Wynikiem kroku 3 była tablica mapująca, które części oryginalnego obrazu są najbardziej interesujące. Ale ta tablica jest nadal dość duża:


 ![](https://miro.medium.com/max/875/1*1WWTbW9yyEJ69TF1rsPv4g.png) 
 
Aby zmniejszyć rozmiar tablicy, zmniejszamy ją za pomocą algorytmu o nazwie max pooling. Brzmi dziwnie, ale wcale tak nie jest. Po prostu przyjrzymy się każdemu kwadratowi 2x2, tablicy i zachowamy największą liczbę

 ![](https://miro.medium.com/max/875/1*xOAroFiw9X0WSkCwgcIO6Q.png) 

Pomysł jest taki, że jeśli znaleźliśmy coś interesującego w którejkolwiek z czterech płytek wejściowych, które tworzą każdy kwadrat siatki 2x2, po prostu zachowamy najbardziej interesujący fragment. Zmniejsza to rozmiar naszej tablicy przy zachowaniu najważniejszych bitów.

##### Krok 5: Przeprowadź prognozę

Do tej pory zredukowaliśmy gigantyczny obraz do dość małej tablicy. Ta tablica to tylko zbiór liczb, więc możemy użyć tej małej tablicy jako danych wejściowych do innej sieci neuronowej. Ta ostateczna sieć neuronowa zdecyduje, czy obraz pasuje, czy nie. Aby odróżnić ją od etapu konwolucji, nazywamy to siecią „w pełni połączoną”.



 |Tak więc od początku do końca cały nasz pięciostopniowy potok wygląda następująco| 
|:---:|
 |![](https://miro.medium.com/max/875/1*tJ1Rkl5xw_5izEZXmNfh5Q.png) |
 
 
 ### Dodawanie jeszcze większej liczby kroków
 
 Nasz potok przetwarzania obrazu składa się z szeregu etapów: konwolucji, max poolingu i wreszcie w pełni połączonej sieci.
Podczas rozwiązywania problemów te kroki można łączyć i układać dowolną liczbę razy! Możma mieć dwie, trzy lub nawet dziesięć warstw knwolucji. Można wprowadzić max pooling w dowolnym miejscu, w którym checmy zmniejszyć rozmiar danych.
Podstawowym pomysłem jest rozpoczęcie od dużego obrazu i ciągłe zmniejszanie go, krok po kroku, aż w końcu uzyskamy pojedynczy wynik. Im więcej jest kroków konwolucyjnych, tym bardziej skomplikowane funkcje będzie w stanie rozpoznać Twoja sieć.
Na przykład, przy wykrywaniu ptaków na zdjęciach, pierwszy krok konwolucji może nauczyć się rozpoznawać ostre krawędzie, drugi krok  może rozpoznawać dzioby na podstawie wiedzy o ostrych krawędziach, trzeci krok może rozpoznawać całe ptaki na podstawie wiedzy o dziobach itp.

 |Oto jak wygląda bardziej realistyczna głęboka sieć konwolucyjna| 
|:---:|
 |![](https://miro.medium.com/max/875/1*JSnKtzEgiHd4p6UlNv_C7w.png) |
 

W tym przypadku uruchamiają obraz o wymiarach 224 x 224 pikseli, stosują konwolucje i dwukrotnie max pooling, stosują konwolucje jeszcze 3 razy, stosują maxpooling, a następnie mają dwie w pełni połączone warstwy. Efekt końcowy jest taki, że obraz jest klasyfikowany do jednej z 1000 kategorii!


### 0 Budowa właściwej sieci  (// w tym rozdziale trzeba coś mądrego napisać)

Skąd wiadomo, które kroki należy połączyć, aby klasyfikator obrazu działał jak najlepiej?
Trzeba na to odpowiedzieć, wykonując wiele eksperymentów i testów. Być może będzie trzeba wytrenować 100 sieci, zanim znajdziemy optymalną strukturę i parametry dla rozwiązywanego problemu. Uczenie maszynowe wymaga wielu prób i błędów.

## Tworzenie klasyfikatora ptaków

#### 1. pataki był w arytkule - trzba to zaznaczyć. bo będziemy robili to co w artykule. cały sens CNN jest wytłumaczony wyżej więc nie ma sensu wymyślać swoich modeli i szukać baz danych

 - to czy wszystko powyżej czyli: wprowadzenie, ML, Deep learing, CNN jest dobrze wytłumaczone to mam do ciebie prośbe jakbyś rzucić okiem co ja tam najebany napisałem - ja to jeszcze przejrze ty tylko zerknij tak bardziej pod względem technicznym - czy nie pierdole głupot (no niestety jestem w stanie jakium jestem i to jest constant - przepraszam bro naprawdę)

# 2. reszta artykułu jest do dokończenia - jakieś 1h pracy z tłumaczeniem (?chyba?)
# 3. trzeba jeszcze zredagować tekst
- poprawić błędy gramatyczne, literówki, 
- zmienić styl językowy na taki bardziej typu sprawozdanie, nie artykułowy 
    - wyjebać teksty takie jak "to bardzo proste musisz tylko coś tam, heh ML is funny bruh") bo mr. Bielecki się może obrazić. 
- trzeba zmienić wyrażenia typu "machine learning", "artificial inteligence", czy "deep learning" na:
    - albo machine learning - ML
    - albo machine learning - uczenia maszynowe
    - tak żeby była spójność językowa - albo skróty, albo po polsku albo po angielsku

# 4. jeszcze jakiś code snippet trzeba dodać (jest na dole artykułu)
# 5. do tego chyba jeszcze testy (tzn. jak się to testuje - ale to też copy paste bo na dole artykułu jest)
# 6. ja chciałbym to przerzucić na web app, tak by mr. Bielecki mógł sobie kilknąć link, zobaczyć sprawozdanie/artykuł (w naszej wersji) razem z imlpementacją. a obok bym jebnął skrypt tak żeby sobie mógł sprawdzić czy to działa i żeby mu wypluwało jakieś dane (jak się da) bo: 
 - mam to napisane w .md ale zapisywanie do pdf nie wychodzi (i nie wiem jak to wszystko mr. Bieleckiemu pokazać - w jakim formacie on to chciał)
    - pisałem to na dillinger.io - chyba nie najepszy wybór
    - za to ładnie mogę pobrać Styled HTML - więc osadzenie tam skryptu powinno być łatwe. i wtedy mamy jeden plik .html jako static web page (chyba, żę import pythona nie pozwala - nie znam się za bardzo)
 - może wtedy mr. Bielecki trochę łaskawszym okiem spojrzy jak dorzucimy jakiś interfejs webowy
 
# 7. nie wiem do kiedy deadline ale taką apkę to chyba w 2 minuty (bo mamy plik .html w którym jest wszystko) i albo dorobić jakiś kod w .js i na githubpages albo tak jak mówiłeś o heroku a jak nie to jebać :D
 
# 8. chyba większość podpunktów załatwie sam tylko trochę czasu potrzebuje (i motywacji xD - ale zapierdalam dzień i noc), z tym web app - to twoja decyzja czy robimy czy tylko oddajemy sprawozdanie w formie: 
-   pdf
-   Styled Html - i podis otwórz sobie to zobaczysz zioooom
-   gdzieś to hostujemy jako jakąś stronke 
-   twoje pomysły

> jak to przeczytasz to daj znać - ja poprawię/zrobię wszystko co trzeba. Ty tylko zadecyduj co trzeba. Fajnie jakbyś zerknął na rzeczy z pkt 6,7,8.ewentualnie pkt 1 może 3


