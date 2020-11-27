# Głębokie uczenie maszynowe i sieci konwolucyjne - znajdowanie na obrazie konkretnego obiektu


W niniejszym sprawdozdaniu zostanie przedstawiony sposób działania głębokiego uczenia maszynowego w połączeniu z sieciami konwolucyjnymi. Przyjrzyżmy się po krótce całemu obszarowi sztucznej inteligencji, uczenia maszynowego, stosowanych technik i podejść oraz znanych zastosowań i ograniczeń. Przy użyciu dostępnych dzisiaj narzędzi "nakarmimy" i zaimplementujemy w prosty sposób program rozpoznający obiekty (ptaki w naszym przypadku) na obrazach przy użyciu wspomnianych technik. Wyjaśnimy, skrótowo i po części, jak aplikacja np. taka jak Google Photos umożliwia wyszukiwanie zdjęć na podstawie tego, co jest na obrazie:


|  Google umożliwia teraz wyszukiwanie własnych zdjęć według opisu - nawet jeśli nie są one otagowane  |
|---|
|![](https://miro.medium.com/max/875/1*F-6upZSC6GMMTP9yHeuwDg.gif)|



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

Istnieje kilka rodzaji operacji Polling'u np. Max pooling i Average Pooling. Max Pooling zwraca maksymalną wartość z części obrazu "pokrytej" przez jądro (kernel). Z drugiej strony, Average Pooling zwraca średnią wszystkich wartości z części obrazu "pokrytej" przez jądro. Najczęściej używany jest Max polling. Dzieli on obraz wejściowy na zestaw nienakładających się prostokątów i, dla każdego takiego podregionu, wyprowadza maksimum.

|(krok/slide = 2, filter 2x2) Max pooling wyciąga największą wartość z kwadratu 2x2 pokrywającage dany obszar w każdym kroku tj. (20, 30, 112, 37). Averge pooling wyciąga średnią wartość w każdeym kroku tj. (13, 8, 79, 20)| 
|:---:|
|![](https://miro.medium.com/max/625/1*KQIEqhxzICU7thjaQBfPBQ.png)|

Warstwa konwolucyjna i warstwa Poolingu tworzą razem i-tą warstwę CNN. W zależności od złożoności obrazów, liczba takich warstw może zostać zwiększona, aby jeszcze bardziej uchwycić szczegóły, ale kosztem większej mocy obliczeniowej.
Po przejściu przez powyższy proces z powodzeniem umożliwiliśmy modelowi zrozumienie cech obrazu wejściowego. Przechodząc dalej, zamierzamy spłaszczyć końcowy output i przesłać go do zwykłej sieci neuronowej w celu klasyfikacji.

### 4. Klasyfikacja - w pełni połączona warstwa (Fully Connected Layer)
Dodanie warstwy w pełni połączonej jest (zwykle) tanim sposobem uczenia się nieliniowych kombinacji cech wysokiego poziomu, reprezentowanych przez dane wyjściowe warstwy konwolucyjnej.



|FC layer| 
|:---:|
|![](https://miro.medium.com/max/875/1*kToStLowjokojIQ7pY2ynQ.jpeg)|


Teraz, gdy przekonwertowaliśmy nasz obraz wejściowy do odpowiedniej postaci dla naszego wielopoziomowego perceptronu, spłaszczamy obraz do wektora kolumnowego. Spłaszczony output jest podawany do sieci neuronowej ze sprzężeniem zwrotnym i propagacją wsteczną stosowaną w każdej iteracji treningu. Na przestrzeni szeregu okresów szkoleniowych model jest w stanie rozróżnić cechy dominujące i pewne cechy niskiego poziomu w obrazach a następnie sklasyfikować je za pomocą techniki klasyfikacji Softmax.


# Rozpoznawanie obrazów - ograniczenia i techniki
***Na początek proste rozpoznawanie pisma odręcznego (liczby osiem)***

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



## Konwolucja - jak działa (uproszczony przykład)
 >Dla wygody (dostępność materiałów zaczerpniętych z artykułu) użycie konwolucyjnych sieci neuronowych, w tym rozdziale, zostanie przedstawione na przykładzie zdjęcia dziecka. 
 ![](https://miro.medium.com/max/875/1*v_06o9d5u4k2lp9cTHQUtg.jpeg)
 >
 
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
Podstawowym pomysłem jest rozpoczęcie od dużego obrazu i ciągłe zmniejszanie go, krok po kroku, aż w końcu uzyskamy pojedynczy wynik. Im więcej jest kroków konwolucyjnych, tym bardziej skomplikowane elementy będzie w stanie rozpoznać nasza sieć.
Na przykład, przy wykrywaniu ptaków na zdjęciach, pierwszy krok konwolucji może nauczyć się rozpoznawać ostre krawędzie, drugi krok  może rozpoznawać dzioby na podstawie wiedzy o ostrych krawędziach, trzeci krok może rozpoznawać całe ptaki na podstawie wiedzy o dziobach itp.

 |Oto jak wygląda bardziej realistyczna głęboka sieć konwolucyjna| 
|:---:|
 |![](https://miro.medium.com/max/875/1*JSnKtzEgiHd4p6UlNv_C7w.png) |
 

W tym przypadku uruchamiają obraz o wymiarach 224 x 224 pikseli, stosują konwolucje i dwukrotnie max pooling, stosują konwolucje jeszcze 3 razy, stosują maxpooling, a następnie mają dwie w pełni połączone warstwy. Efekt końcowy jest taki, że obraz jest klasyfikowany do jednej z 1000 kategorii!

Skąd wiadomo, które kroki należy połączyć, aby klasyfikator obrazu działał jak najlepiej?
Odpowiedzieć na to pytanie, wykonując wiele eksperymentów i testów?. Być może będzie trzeba wytrenować 100 sieci, zanim znajdziemy optymalną strukturę i parametry dla rozwiązywanego problemu. Dodatkowo oprócz opisanych w tym rozdziale elementów istnieją jeszcze inne udoskonalenia CNN. Dostępne i dobrze znane są różne architektury CNN. Odegrały one kluczową rolę w tworzeniu algorytmów, które zasilają i będą zasilać sztuczną inteligencję jako całość w dającej się przewidzieć przyszłości. Niektóre z nich to:
- LeNet
- AlexNet
- VGGNet
- GoogLeNet
- ResNet
- ZFNet

# Praktyczny przykład wraz z implementacją (tensorflow)

>"Klasyfikator ptaków"


Jak w każdym przypadku potrzebujemy danych, aby rozpocząć. Bezpłatny zestaw danych [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) zawiera 6000 zdjęć ptaków i 52000 zdjęć rzeczy, które nie są ptakami. Aby jednak uzyskać jeszcze więcej danych, dodamy również zestaw danych [Caltech-UCSD Birds-200–2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) , który zawiera kolejne 12 000 zdjęć ptaków. 

|Oto kilka ptaków z naszego połączonego zbioru danych| 
|:---:|
|![](https://miro.medium.com/max/1250/1*r9I5D3NXCn8gnLOjahuSQA.png) |
 
|A oto niektóre z 52 000 obrazów innych niż ptaki| 
|:---:|
|![](https://miro.medium.com/max/1250/1*ODaXoLQY4-D7zqHrqeA4Uw.png) |


Ten zestaw danych będzie działał dobrze w naszym prostym przykładzie, ale 72 000 obrazów o niskiej rozdzielczości to wciąż dość mało dla rzeczywistych zastosowań. Jeśli zależałoby nam na wydajności na poziomie aplikacji Google, potrzeba by nam milionów dużych obrazów. W uczeniu maszynowym posiadanie większej ilości danych jest prawie zawsze ważniejsze niż lepsze algorytmy.


Aby zbudować nasz klasyfikator, użyjemy [TFLearn](http://tflearn.org/). TFlearn to wrapper biblioteki  [TensorFlow](https://www.tensorflow.org/) od Google. Służy  do głębokiego uczenia, oraz udostępnia uproszczony interfejs API, dzięki czemu budowanie konwolucyjnych sieci neuronowych jest bardzo proste.

Oto kod definiujący i szkolący sieć:

```python
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Load the data set
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

# Shuffle the data
X, Y = shuffle(X, Y)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='bird-classifier')

# Save model when training is complete to a file
model.save("bird-classifier.tfl")
print("Network trained and saved as bird-classifier.tfl!")
```

 
Jeśli trenujemy sieć z dobrą kartą graficzną z wystarczającą ilością pamięci RAM, zostanie to zrobione w mniej niż godzinę. Jeśli używamy zwykłego CPU, może to zająć dużo więcej czasu.

W miarę treningu dokładność wzrasta. Po pierwszym przejściu ok. 75,4% dokładności. Po 10 przejściach był to poziom ok. 91,7%. Po około 50 cyklach dokłądność osiągnęła poziom 95,5% dokłądności. Dodatkowe szkolenie nie przynosiło znaczących popraw. Teraz nasz program potrafi teraz rozpoznawać ptaki na obrazach!


# TODO
> info brałem z tego co podlinkowałeś https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721
>
> z tego trochę https://en.wikipedia.org/wiki/Convolutional_neural_network#Building_blocks
>

> i z tego trochę https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

# 1. Trzeba sprawdzić czy to co napisałem jest w całości potrzebne. 
 - Część teoretyczna czyli wprowadadzenie i wytłumaczenia terminów: Uczenie maszynowe, Uczenie głębokie, Sieci neuronowe, CNN (tutaj patrzyłem na wikipedie i inne artykuły) 
 - Następnie jest "Rozpoznawanie obrazów" na podstawie pisma odręcznego (to jest prosto z artykułu)
 - "Konwolucja - jak działa (uproszczony przykład)" - gwóźdź programu (to też prosto z artykułu)
 - Trzeba dokończyć tłumaczenie artykułu - od tego rozdziału "Building our Bird Classifier" do "Testing our Network"
 

# 2. Trzeba sprawdzić czy to wszystko powyżej ma sens (albo nie, może mr. Bielecki tego nie będzie czytać, bo mi by się nie chciało)


# 3. trzeba  zredagować tekst
- poprawić błędy gramatyczne, literówki, 
- zmienić styl językowy na taki bardziej typu sprawozdanie, nie artykułowy 
    - wyjebać teksty takie jak "to bardzo proste musisz tylko coś tam, heh ML is funny bruh") bo mr. Bielecki się może obrazić. 
- trzeba zmienić wyrażenia typu "machine learning", "artificial inteligence", czy "deep learning" na:
    - albo machine learning - ML
    - albo machine learning - uczenia maszynowe
    - tak żeby była spójność językowa - albo skróty, albo po polsku albo po angielsku


# 4. trzeba to ułożyć w jakiejś sensownej kolejności. Albo może tak zostać - whatever



 




