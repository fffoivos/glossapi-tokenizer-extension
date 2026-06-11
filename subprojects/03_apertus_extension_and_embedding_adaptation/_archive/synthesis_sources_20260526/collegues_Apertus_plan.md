Apertus-Greek: 
Βελτίωση του μοντέλου στα Ελληνικά

Το Apertus της Swiss-AI αποτελεί ένα από τα πιο «διαφανή» μοντέλα αυτή τη στιγμή (Fully documented architecture). Το γεγονός ότι βασίζεται στο FineWeb2-HQ του προσφέρει μια εξαιρετικά στιβαρή βάση.Το FineWeb2-HQ παρέχει υψηλή πυκνότητα πληροφορίας (information density) στα αγγλικά, γεγονός που επιτρέπει την αποτελεσματική μεταφορά γνώσης (transfer learning) μέσω CPT
Ωστόσο, τα γενικά σύνολα δεδομένων (datasets), όπως το FineWeb, συχνά υστερούν σε εξειδικευμένη εντοπιότητα (π.χ. ελληνική νομοθεσία, ιατρική ορολογία, λογοτεχνία ή εξαιρετικά τεχνικά κείμενα).
Σε αυτο το άρθρο εντοπίζουμε τα «κενά» γνώσης του Apertus και επιλέγουμε τα κατάλληλα τμήματα από τα σύνολα δεδομένων του GlossAPI για την περαιτέρω εκπαίδευση του μοντέλου, ώστε να καταστεί πιο αποδοτικό σε τομείς που ενδιαφέρουν τους Έλληνες χρήστες.
Αξιολόγηση Ελληνικών Δεδομένων glossApi

Μεθοδολογία
Για να εντοπίσουμε τι «λείπει» από το Apertus, πρέπει να μετρήσουμε τον βαθμό «έκπληξης» (probability distribution mismatch) του μοντέλου όταν έρχεται αντιμέτωπο με ελληνικά δεδομένα.
Περπλεξία (Perplexity)
Μαθηματικά, η περπλεξία ορίζεται ως:
$$PPL(X) = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i | x_{<i}) \right)$$

Περπλεξία είναι η εκθετική μορφή του μέσου όρου της απώλειας από την διασταυρούμενη εντροπία και μετράει πόσο καλά το μοντέλο προβλέπει μια ακολουθία λέξεων. Χαμηλότερη περπλεξία σημαίνει καλύτερη πρόβλεψη και άρα καλύτερη γνώση του μοντέλου για το συγκεκριμένο κείμενο.
Αυτός είναι ο πλέον αντικειμενικός τρόπος μέτρησης. Λαμβάνουμε δείγματα (π.χ. 500-1000 γραμμές) από κάθε επιμέρους σύνολο δεδομένων του glossAPI και υπολογίζουμε την Περπλεξία (Perplexity - PPL) του Apertus σε αυτά.

Αν εχει καποιο dataset έχει Υψηλή Περπλεξία το μοντέλο «δυσκολεύεται» να προβλέψει τις λέξεις, άρα παρουσιάζει κενό γνώσης στον συγκεκριμένο τομέα. 
 Αυτά τα σύνολα δεδομένων πρέπει να ενταχθούν στη διαδικασία Συνεχούς Προ-εκπαίδευσης (Continual Pre-training - CPT).
 Όμως προσοχή υψηλή περπλεξία μπορεί επίσης να σημαίνει κακή ποιότητα δεδομένων (garbage in, garbage out) ή ασυμβατότητα του υπάρχοντος tokenizer με το domain. Πρέπει να διευκρινιστεί ότι η υψηλή PPL είναι ένδειξη για CPT μόνο αν η ποιότητα (Quality Score) είναι ταυτόχρονα υψηλή.


Με Χαμηλή Περπλεξία  το μοντέλο είναι ήδη εξοικειωμένο με το ύφος και το περιεχόμενο, επομένως ενδέχεται να μη χρειάζεται επιπλέον εκπαίδευση.


Αποτελέσματα εργαλείου
Για τον ελεγχο των dataset αναπτύξαμε το δικό μας εργαλείο ελέγχου ωστε να παράξουμε τοσο οπτικοποιημένη την περπλεξια όσο και αποτελεσματα μετρησης σε όλα τα διαθέσιμα datasets.


Στα παρακάτω διαγράμματα κάθε σημείο αντιπροσωπεύει την ενσωμάτωση ενός εγγράφου. Το χρώμα αντιστοιχεί στο αναγνωριστικό του συνόλου δεδομένων (dataset_id). Τα δείγματα αναφοράς σημειώνονται με X, ενώ τα υποψήφια σύνολα με κύκλο.

Τα δεδομενα μπορει να έχουν κάποια από τις τρεις παρακάτω ιδιότητες:


Θεματική Εγγύτητα: Αν ένα υποψήφιο σύνολο δεδομένων συμπίπτει με το "νέφος" του FineWeb2-HQ, πιθανότατα περιέχει πλεονάζουσα (redundant) πληροφορία.

Καινοτομία (Novelty): Αν ένα σύνολο σχηματίζει ξεχωριστή συστάδα, αυτό υποδηλώνει μετατόπιση της κατανομής (distribution shift) ή θεματική καινοτομία — θετικό στοιχείο για CPT.

Θόρυβος: Μεμονωμένες ακραίες τιμές (outliers) που δεν σχηματίζουν συστάδα συνήθως υποδεικνύουν θόρυβο, προβλήματα μορφοποίησης (OCR) ή τυποποιημένα κείμενα (boilerplate), και όχι χρήσιμη γνώση.






PCA


Το PCA αναδεικνύει τις κύριες γραμμικές διακυμάνσεις. Χρησιμοποιείται για τον εντοπισμό καθολικών θεματικών αποκλίσεων από το FineWeb2-HQ baseline Ειναι δηλαδή μια μακροσκοπική εικόνα της θεματικής κατανομής.

t-SNE



Το t-SNE εστιάζει στις τοπικές γειτονιές. Χρησιμοποιείται για τον εντοπισμό clusters με ειδική ορολογία που ο tokenizer αδυνατεί να συμπιέσει σωστά. Είναι δηλαδή ιδανικό για να δούμε ποια δείγματα μοιάζουν μεταξύ τους σε πολύ συγκεκριμένο επίπεδο. 





Πρακτικός κανόνας: Ο συνδυασμός μεγάλης απόστασης από τη βάση αναφοράς, υψηλής περπλεξίας και αποδεκτής ποιότητας κειμένου αποτελεί την ισχυρότερη ένδειξη για έναν καλό υποψήφιο CPT.








							
								
						

							

						

Ranking Table
HP=high priority,LP=low priority or redundant,NR=needs manual review,HPG=high perplexity gap,LSO=low semantic overlap,AQP=acceptable quality profile,NR=noise risk,MPM=missing perplexity measurement,ISD=insufficient signal diversity


Dataset
Bucket
Priority Score
Mean PPL
Quality
Gap
Novelty
Noise Penalty
Rationale
glossAPI/modern-greek-dictionary
HP
2.480
21.707
0.642
1.000
0.856
0.018
HPG, LSO
glossAPI/artos-zois
HP
1.864
7.079
0.997
0.215
0.652
0.000
AQP
glossAPI/Ellinika_Keimena_Project_Gutenberg
HP
1.813
8.255
0.997
0.266
0.550
0.000
AQP
glossAPI/klasikh_arx_ell_grammateia
HP
1.754
4.759
0.997
0.107
0.650
0.000
AQP
glossAPI/eurlex-greek-legislation
HP
1.742
2.523
0.992
0.000
0.750
0.000
LSO, AQP
glossAPI/Wikisource_Greek_texts
HP
1.742
6.931
0.968
0.216
0.558
0.000
AQP
glossAPI/Ekklisiastika_Keimena
HP
1.733
4.342
0.992
0.082
0.659
0.000
AQP
glossAPI/archetai
HP
1.727
7.461
0.972
0.242
0.547
0.035
AQP
glossAPI/dimodis_logotexnia
HP
1.721
4.363
0.999
0.084
0.637
0.000
AQP
glossAPI/1000_prwta_xronia_ellhnikhs
HP
1.709
4.471
0.983
0.093
0.654
0.021
AQP
glossAPI/opengov-deliberations-v2
HP
1.516
3.147
0.712
0.028
0.781
0.005
LSO, AQP
glossAPI/openarchives.gr
HP
1.489
7.243
0.680
0.219
0.589
0.000
AQP
glossAPI/Sxolika_vivlia
HP
1.484
3.699
1.000
0.053
0.431
0.000
AQP
glossAPI/ert-press
HP
1.472
4.392
0.757
0.086
0.628
0.000
AQP
glossAPI/95k_deigma_ellinikis
HP
1.422
6.936
0.650
0.216
0.556
0.000
ISD
glossAPI/istorima
HP
1.419
5.046
0.676
0.116
0.627
0.000
AQP
glossAPI/openbook.gr
LP
0.778
5.513
0.716
0.145
0.340
0.423
AQP
glossAPI/eellak-articles
LP
0.200
4.987
0.245
0.129
0.659
0.834
noise risk
glossAPI/Greek_PhD_Theses_Corpus
NR
1.631
NA
0.968
0.000
0.663
0.000
missing perplexity measurement, LSO, AQP
glossAPI/Apothetirio_Kallipos
NR
1.376
4.173
0.701
0.077
0.831
0.233
LSO, AQP
glossAPI/e-nautilia
NR
1.139
3.659
0.787
0.057
0.616
0.320
AQP
glossAPI/Apothetirio_Pergamos
NR
1.109
3.758
0.689
0.057
0.462
0.098
AQP
glossAPI/amna-press
NR
1.018
3.621
0.660
0.050
0.395
0.087
AQP
glossAPI/ellinika_dedomena_europaikou_koinovouliou
NR
1.014
3.143
0.619
0.034
0.682
0.321
LSO



Άρκετά σύνολα δεδομένων έχουν noise penalty 0 επειδή, με τις heuristics που έτρεξαν, μοιάζουν καθαρά ως προς markup/OCR/boilerplate. Αυτό δεν σημαίνει ότι είναι τέλεια σύνολα δεδομένων σε κάθε έννοια ποιότητας. Σημαίνει μόνο ότι δεν ενεργοποιούν αυτά τα τρία συγκεκριμένα noise signals.

Επικρατέστερη λίστα για CPT
Για το επερχόμενο CPT διατηρούμε σύνολα δεδομένων που συνδυάζουν έντονη καινοτομία είτε με ένα σαφές κενό αμηχανίας (αδυναμίας κατανόησης του Apertus)  είτε με πολύ ισχυρή ποιότητα.

Τελικό καθαρισμένο dataset: glossapi-greek-nanochat-pretraining-dataset https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset/settings




Dataset
Priority Score
Mean PPL
Quality
Gap
Novelty
Noise Penalty
Why Keep It
glossAPI/modern-greek-dictionary
2.480
21.707
0.642
1.000
0.856
0.018
Strongest gap signal in the run plus very high novelty
glossAPI/artos-zois
1.864
7.079
0.997
0.215
0.652
0.000
Balanced profile with high quality and meaningful gap
glossAPI/Ellinika_Keimena_Project_Gutenberg
1.813
8.255
0.997
0.266
0.550
0.000
High-quality literary corpus with clear model gap
glossAPI/Ekklisiastika_Keimena
1.733
4.342
0.992
0.082
0.659
0.000
High quality plus strong thematic novelty
glossAPI/1000_prwta_xronia_ellhnikhs
1.709
4.471
0.983
0.093
0.654
0.021
Strong novelty with low noise and stable quality
glossAPI/eurlex-greek-legislation
1.742
2.523
0.992
0.000
0.750
0.000
Specialized legal domain with excellent quality and very strong novelty
glossAPI/openarchives.gr


1.489
7.243
0.680
0.219
0.589
0.000
Balanced academic and archive coverage with real model gap
Greek_PhD_Theses_Corpus
























glossAPI/modern-greek-dictionary βγαίνει πολύ ψηλά επειδή έχει μακράν το ισχυρότερο gap signal. Το mean PPL είναι 21.707 και το gap score 1.000, ενώ η novelty είναι επίσης πολύ υψηλή στο 0.858. Η quality είναι μόνο μέτρια στο 0.642, αλλά το noise penalty είναι χαμηλό, άρα το ranking το διαβάζει ως πραγματικό knowledge/domain miss και όχι ως σκουπίδι.


glossAPI/eurlex-greek-legislation βγαίνει ψηλά για σχεδόν αντίθετο λόγο. Το PPL του είναι χαμηλό, άρα δεν φωνάζει model weakness, αλλά έχει πολύ υψηλή quality, 0.992, και ισχυρή novelty, 0.676. Άρα είναι καλός domain-specific CPT candidate, όχι gap-repair candidate.

glossAPI/openarchives.gr είναι το πιο ισορροπημένο από τα τρία. Έχει πραγματικό gap, mean PPL 7.243 και gap score 0.219, ισχυρή novelty, 0.663, αποδεκτή quality, 0.680, και μηδενικά redundancy/noise penalties. Αυτό το κάνει καλό candidate για χρήσιμο νέο υλικό, ειδικά τώρα που το sampling πατάει σωστά στα abstracts.



TOKENIZER (WIP) 
* θα πρέπει να γινει επαναπροσδιορισμός σε σχεση με τα επιλεγμενα dataset's τα παρακατω ειναι απο test χωρίς προσθήκη νέας γνώσης.

Λογική Υλοποίησης και Επέκτασης του Tokenizer
Στόχος αυτού του σταδίου είναι η προσαρμογή του υπάρχοντος tokenizer (swiss-ai/Apertus-8B-Instruct-2509) ώστε να υποστηρίζει καλύτερα την ελληνική γλώσσα, να μειώσει τον κατακερματισμό (fragmentation) των ελληνικών κειμένων σε πολλά υπο-τμήματα (subtokens) και να βελτιστοποιήσει την αναπαράσταση ειδικής ορολογίας, όπως αυτής του GlossAPI.

Η διαδικασία χωρίζεται στα εξής βασικά βήματα
Εξαγωγή Στατιστικών Λέξεων από Κείμενα (Word Statistics Extraction)
Αρχικά, αναλύεται ένα μεγάλο ελληνικό corpus (όπως το ελληνικό τμήμα του FineWeb2-HQ) για την εύρεση της συχνότητας των λέξεων.
Αυτό γίνεται μέσω του εργαλείου vocabularyGen/countWords.py το οποίο με ένα πέρασμα (streaming):
Μετράει την εμφάνιση απλών λέξεων (words).
Εξάγει λέξεις που βρίσκονται μέσα σε χωρία με εισαγωγικά (quoted words).
Καταγράφει λέξεις που ξεκινούν με κεφαλαίο (capitalized words: ονόματα, χώρες, κλπ.). Τα δεδομένα καταγράφονται σε βάσεις δεδομένων SQLite, ώστε να αποφευχθεί το υπερβολικό φόρτωμα της μνήμης (RAM) από το τεράστιο μέγεθος των corpus.
Επιλογή Υποψήφιων Tokens (Candidate Selection)
Η εισαγωγή νέου λεξιλογίου δεν γίνεται αδιάκριτα. Γίνεται χρήση του εργαλείου vocabularyGen/selectTokenizerCandidates.py για την ακριβή επιλογή:
Ανάλυση Κατακερματισμού: Ελέγχεται πώς ο αρχικός (base) tokenizer τεμαχίζει την κάθε υποψήφια λέξη σε subtokens.
Φιλτράρισμα: Επιλέγονται λέξεις που εμφανίζονται αρκετά συχνά στα δεδομένα αλλά ταυτόχρονα κατακερματίζονται πολύ από τον παλιό tokenizer (π.χ. χρειάζονται 4-5 tokens). Λέξεις με τεράστια συχνότητα περνούν από αυστηρότερα κριτήρια για να μην χαλάσει η ισορροπία του γενικού λεξιλογίου.
Ενοποίηση Τύπων: Γίνεται συγχώνευση (case-folding) κεφαλαίων/πεζών αναλογικά με τις ανάγκες, εκτός αν ρυθμιστεί διαφορετικά.
Curated Στατικά Tokens: Πέρα από τα tokens του corpus, προστίθενται στατικές λίστες από curated tokens σημαντικά για το domain-specific περιβάλλον μας (GlossAPI). Το τελικό αρχείο επιλεγμένων λέξεων (selected_tokens_v1.txt) περιλαμβάνει πλέον τις λέξεις συνήθως με ένα αρχικό κενό διάστημα, ώστε να ταιριάζουν στα όρια (word boundaries) του tokenizer.
Επέκταση Tokenizer και Έξυπνη Αρχικοποίηση (Alignment & Initialization)
Έχοντας το επιλεγμένο λεξιλόγιο, προχωράμε στην επέκταση:
Μέσω του scripts/extend_apertus_tokenizer.py προσθέτουμε τα νέα tokens στον base tokenizer, δημιουργώντας τη νέα έκδοση (π.χ. apertus-greek-v1).
Resizing Model Embeddings: Το ίδιο το μοντέλο φορτώνεται για να αυξηθεί το μέγεθος του πίνακα των embeddings ώστε να περιλάβει το νέο αυξημένο μέγεθος του λεξιλογίου.
Υπάρχουν διάφορες τεχνικές για τρόπο αρχικοποίησης των νέων token. Θεωρώ πως  mean init είναι στις τεχνικές όπου έχουν πολύ μεγάλη επίρροη οι έννοιες από ξένες γλώσσες πάνω στην αρχικοποίηση των Ελληνικών, θέλουμε τεχνικές που αξιοποιούν τα ως τώρα εκπαιδευμένα tokens στα Ελληνικά για να αρχικοποιήσουν τα υπόλοιπα, ώστε η κύρια διαμόρφωση να είναι από τα Ελληνικά δεδομένα εκπαίδευσης. Πρέπει να έχουμε υπόψη μας όμως πως η εκπαίδευση χωρίς επέκταση του λεξιλογίου μπορεί να έχει εξίσου καλά ή και καλύτερα αποτελέσματα ως προς την αφομοίωση των δεδομένων μας, και το κύριο πλεονέκτημα της επέκτασης θα είναι αποτελεσματικότητας/οικονομικό. Άρα τα τρία βασικά πειράματα που προτείνω είναι: [Δες αναλυτικά Apertus Documentation - Greek Implementation V2 Draft]
Vanilla: εκπαίδευση χωρίς επέκταση.
Mean Initialization: Τα καινούρια tokens δεν αρχικοποιούνται από το μηδέν ή με τυχαίο θόρυβο. Το script ελέγχει τα subtokens που θα δίνονταν προηγουμένως για τη συγκεκριμένη λέξη, εξάγει τα embeddings τους από το αρχικό μοντέλο, υπολογίζει τον μέσο όρο τους, και τα τοποθετεί στο νέο, ενιαίο token. Αυτή η πρακτική μεταφέρει μερική γνώση της έννοιας εξαρχής στο νέο token. Όμως προσοχή αν τα αρχικά subtokens έχουν χαμηλή συχνότητα στο base model, ο μέσος όρος τους θα είναι θόρυβος. 
ReTok: στο ReTok αξιοποιείται το merge table του BPE για την αρχικοποίηση  κάθε νέου token με E_new[T] = (E[L] + E[R]) / 2 , όπου E[L] και E[R] τα embeddings του  αριστερού και δεξιού token.
Token Distillation: ξεκινάει με ReTok, μετά βρίσκοντας σημείο όπου παλιά και καινούργια tokens έχουν υπερκάλυψη (πχ επι-στή-μη = επιστήμη) παγώνει ένα προ-επιλεγμένο attention layer σε αυτό το σημείο, και κάνει gradient descent μόνο με τα embeddings ενεργά ώστε να μοιάσει όσο γίνεται η ενεργοποίηση του attention μετά το νέο token με την ενεργοποίηση μετά των παλιών. Είναι πολύ πιο ελαφρύ ως μέθοδος από τη κανονική εκπαίδευση.
Το αποτέλεσμα είναι η δημιουργία ενός σταθερού (persistent) aligned αρχικού checkpoint (μοντέλο και εκτεταμένος tokenizer), το οποίο μπει στην αναμονή, έτοιμο για τα μεταγενέστερα εκπαιδευτικά στάδια.

Επεξήγηση προσέγγισης:
Η προσέγγιση που περιγράφεται παραπάνω ακολουθεί τις βέλτιστες πρακτικές για την επέκταση λεξιλογίου σε μεγάλα γλωσσικά μοντέλα (LLMs). 

Οι λόγοι που καθιστούν αυτή την προσέγγιση σωστή και ασφαλή είναι:

Στοχευμένη Επιλογή αντί για Τυφλή Εισαγωγή: Η χρήση συχνότητας σε συνδυασμό με τον δείκτη κατακερματισμού (fragmentation) διασφαλίζει ότι προσθέτετε μόνο tokens που πραγματικά προσφέρουν αξία (compression), αποφεύγοντας το "φούσκωμα" του λεξιλογίου με σπάνιες λέξεις που απλώς θα καθυστερήσουν την εκπαίδευση.

Mean Initialization (Έξυπνη Αρχικοποίηση): Αντί να ξεκινήσουν τα νέα embeddings από το μηδέν (ή με τυχαίες τιμές), ο υπολογισμός του μέσου όρου των αρχικών subtokens μεταφέρει την ήδη υπάρχουσα "κατανόηση" του μοντέλου για αυτήν τη λέξη στα νέα ενιαία tokens. Αυτό μειώνει δραματικά το αρχικό σοκ (loss spike) όταν το μοντέλο ξεκινήσει την εκπαίδευση (CPT).

Σταδιακή και Δεδομενοκεντρική Προσέγγιση: Η χρήση SQLite για την καταγραφή πραγματικών συχνοτήτων από το corpus (FineWeb2-HQ) και η εφαρμογή περιπτώσεων ρουτίνας (case-folding) αποτρέπει θόρυβο και διασφαλίζει ότι το dataset εκπαίδευσης συνάδει με το tokenizer.

Είναι ίσως ο πιο επαγγελματικός τρόπος να εισαχθεί μια νέα γλώσσα ή ειδική ορολογία σε ένα ήδη εκπαιδευμένο μοντέλο (όπως το Apertus) πριν ξεκινήσει το Continued Pre-Training (CPT).

Σχέδιο Εκτέλεσης CPT
Συνεχής Προ-εκπαίδευση (Continued Pre-Training - CPT)

Πεδίο Εφαρμογής
Αυτό το σχέδιο βασίζεται στη λίστα από την παραπάνω κατάταξη. Έχει ως στόχο το επόμενο πέρασμα CPT ότι θα πρέπει να βελτιστοποιήσει τρία στοιχεία ταυτόχρονα:
Διατήρηση υψηλής ποιότητας προσαρμογής του ελληνικού πεζού λόγου.
Προσθήκη θεματικής καινοτομίας (domain novelty) που παραμένει χρήσιμη κατά τον χρόνο συμπερασμού (inference).
Αποκατάσταση των μεγαλύτερων μετρούμενων γνωστικών και λεξιλογικών κενών, χωρίς να επιτραπεί σε δομικά ασυνήθιστα δεδομένα να κυριαρχήσουν πολύ νωρίς.
Όλα τα βάρη παρακάτω είναι ποσοστά επί του συνολικού προϋπολογισμού tokens του CPT και αθροίζονται στο 100%.

Μείγμα Εκπαίδευσης και "English Anchor"
Για να αποφύγουμε την απώλεια ικανοτήτων συλλογισμού (catastrophic forgetting), το μείγμα δεδομένων περιλαμβάνει έναν "άγκυρα" αγγλικών δεδομένων. Μια αποδεδειγμένη αναλογία είναι:
90% Ελληνικά κείμενα (από τα dataset)
10% Αγγλικά κείμενα (από το FineWeb-HQ) Με αυτόν τον τρόπο το μοντέλο ενσωματώνει την ελληνική γλώσσα χωρίς να αλλοιώνεται η βασική του λογική δομή.

Αν το μοντέλο αρχίσει να «παραληρεί» (hallucinations) σε logical reasoning tasks, το ποσοστό αυτό πρέπει να ανέβει στο 15-20%.

Προτεινόμενη Σειρά Εκτέλεσης
Προτιμώμενη προσέγγιση: χρήση ενός «ελαφρού προγράμματος σπουδών» (light curriculum) αντί για την εισαγωγή ολόκληρου του μείγματος στο μοντέλο από το βήμα 0.
Βήμα
Σύνολο Δεδομένων (Dataset)
Ρόλος στο Curriculum
Γιατί μπαίνει εδώ


OPUS__OpenSubtitles






HPLT dedublicate






openbook
split to class






2
glossAPI/Ellinika_Keimena_Project_Gutenberg
θεμέλιος πεζός λόγος
Υψηλής ποιότητας λογοτεχνικά Ελληνικά με πραγματικό σήμα κενού


Sxolika_vivlia


dataset per contex




5
glossAPI/openarchives.gr
διεύρυνση τομέα
Ισορροπημένο ακαδημαϊκό/αρχειακό σήμα με πραγματικό κενό και αποδεκτή ποιότητα


Apothetirio_Pergamos






Apothetirio_Kallipos






4
glossAPI/1000_prwta_xronia_ellhnikhs




θεμέλια καινοτομία
Ιστορικό εύρος και καινοτομία με χαμηλό «θόρυβο»


klasikh_arx_ell_grammateia.parquet




3
glossAPI/Ekklisiastika_Keimena
θεμέλια καινοτομία
Ισχυρή ποιότητα με ιδιαίτερο ύφος (register) και χαμηλή σημασιολογική επικάλυψη


Greek_PhD_Theses_Corpus






dimodis_logotexnia






6
glossAPI/eurlex-greek-legislation



εξειδίκευση τομέα
Εξαιρετική ποιότητα και καινοτομία στον νομικό τομέα, αλλά το χαμηλό PPL σημαίνει ότι δεν χρειάζεται να κυριαρχήσει νωρίς


AI-team-UoA__greek_legal_code.






1
glossAPI/artoszois
θεμέλιος πεζός λόγος
Πολύ υψηλή ποιότητα, σημαντικό κενό και καθαρό φυσικό κείμενο το καθιστούν ασφαλές αρχικό στήριγμα
7
glossAPI/modern-greek-dictionary
αποκατάσταση κενών (τελικό στάδιο)
Το υψηλότερο μετρούμενο κενό και υψηλή καινοτομία, αλλά η δομή τύπου λεξικού πρέπει να εισαχθεί αργά και ελεγχόμενα



Προτεινόμενες Φάσεις Εκπαίδευσης
Φάση 1: Βασικό Πέρασμα (Foundation Pass)
Χρήση των καθαρότερων σωμάτων κειμένου συνεχούς ροής (running-text):
glossAPI/artoszois
glossAPI/Ellinika_Keimena_Project_Gutenberg
glossAPI/Ekklisiastika_Keimena
glossAPI/1000_prwta_xronia_ellhnikhs
Στόχος: Σταθεροποίηση σε υψηλής ποιότητας ελληνικό πεζό λόγο και διεύρυνση της κάλυψης ύφους και επιπέδου γλώσσας πριν από την προσθήκη πιο εξειδικευμένου ή δομικά ασυνήθιστου υλικού.
Φάση 2: Διεύρυνση Τομέα (Domain Broadening)
Προσθήκη των ειδικών ανά τομέα, αλλά ακόμα φυσικών συνόλων δεδομένων:
glossAPI/openarchives.gr
glossAPI/eurlex-greek-legislation
Στόχος: Έγχυση ακαδημαϊκής και νομικής κάλυψης αφού το μοντέλο έχει ήδη εδραιωθεί σε καθαρό ελληνικό κείμενο.
Φάση 3: Αποκατάσταση Κενών και Λεξιλογική Πύκνωση
Εισαγωγή του συνόλου δεδομένων με βαρύ λεξιλόγιο στο τέλος:
glossAPI/modern-greek-dictionary
Στόχος: Εκμετάλλευση του πολύ ισχυρού σήματος κενού χωρίς να επιτραπεί στη μορφοποίηση του λεξικού να παραμορφώσει την πρώιμη τροχιά βελτιστοποίησης.
Smoke Tests & Targeted CPT Probe
Η διαδικασία εκπαίδευσης πρέπει να προσπελάσει πρώτα μια "πύλη" αξιολόγησης (Validation Gate):

Δημιουργία Targeted Probe Dataset: Αρχικά εξάγουμε ένα μικρό dataset (π.χ. 1GB από curated υλικό GlossAPI) και τρέχουμε ένα πολύ σύντομο CPT (έως ~100 steps).
Αξιολόγηση Smoke Test: Το μοντέλο δοκιμάζεται σε benchmarks (με το GreekMMLU). Αν η απόδοση πέσει αξιοσημείωτα (regression), η προσέγγιση ελέγχεται ξανά (δεν προχωράμε στο Production). Αν η απόδοση μείνει σταθερή ή βελτιωθεί, παίρνουμε το πράσινο φως.



Προτεινόμενο Τελικό Μείγμα CPT
Για ένα τελικό στατικό μείγμα για την κύρια εκτέλεση του CPT, θα έχουμε αυτή την κατανομή:
Σύνολο Δεδομένων
Προτεινόμενο Βάρος
Αιτιολόγηση Βάρους
glossAPI/modern-greek-dictionary
18%
Το μεγαλύτερο κενό και καινοτομία, αλλά με όριο κάτω από το 20% λόγω δομικής ιδιαιτερότητας
glossAPI/artoszois
16%
Εξαιρετική ποιότητα και σημαντικό κενό· καλό στήριγμα για όλη τη διάρκεια της εκτέλεσης
glossAPI/Ellinika_Keimena_Project_Gutenberg
14%
Υψηλής ποιότητας μακροσκελή Ελληνικά με ισχυρότερο σήμα κενού από το μέσο όρο
glossAPI/openarchives.gr
15%
Ισχυρή ισορροπία κενού και καινοτομίας χωρίς ποινές πλεονασμού/θορύβου
glossAPI/Ekklisiastika_Keimena
12%
Η υψηλή ποιότητα και το ιδιαίτερο ύφος δικαιολογούν ένα σταθερό μεσαίο μερίδιο
glossAPI/1000_prwta_xronia_ellhnikhs
12%
Παρόμοιος ρόλος με τα Εκκλησιαστικά: ισχυρή καινοτομία, χαμηλός θόρυβος και καλή ποιότητα
glossAPI/eurlex-greek-legislation
13%
Πολύ καθαρό εξειδικευμένο νομικό σήμα, αλλά διατηρείται χαμηλότερα γιατί το κενό PPL είναι ήδη μικρό


Πρακτικό Σκεπτικό
Το glossAPI/modern-greek-dictionary λαμβάνει το μεγαλύτερο μεμονωμένο μερίδιο επειδή το κενό είναι πολύ μεγάλο για να αγνοηθεί, αλλά όχι υπερβολικό ποσοστό επειδή το dataset δεν μοιάζει με συνηθισμένο συνεχή πεζό λόγο.
Τα glossAPI/artoszois και glossAPI/Ellinika_Keimena_Project_Gutenberg αποτελούν μαζί την ασφαλέστερη βάση υψηλής ποιότητας για το μείγμα.
Το glossAPI/openarchives.gr έχει σταθμιστεί αρκετά επιθετικά ώστε να έχει σημασία, καθώς συνεισφέρει τόσο καινοτομία όσο και πραγματική δυσκολία για το μοντέλο.
Το glossAPI/eurlex-greek-legislation παραμένει σημαντικό στο μείγμα λόγω της σημασίας της εξειδίκευσης, αλλά δεν χρειάζεται το ίδιο βάρος με τα σύνολα δεδομένων που προορίζονται για καθαρή αποκατάσταση κενών.
Τα glossAPI/Ekklisiastika_Keimena και glossAPI/1000_prwta_xronia_ellhnikhs βοηθούν στο να μην περιοριστεί η λίστα σε μία μόνο οικογένεια ύφους.
Δικλείδες Ασφαλείας (Guardrails)
Εάν τα πρώτα checkpoints αρχίσουν να φαίνονται υπερβολικά σαν γλωσσάρια ή λίστες, θα μειώσουμε το glossAPI/modern-greek-dictionary από 18% σε 12-15% και θα ανακατανείμουμε  τη διαφορά στα glossAPI/artoszois και glossAPI/openarchives.gr.
Εάν η βελτίωση στον νομικό τομέα χρειάζεται να βελτιωθει, θα μεταφέρουμε 2-3 μονάδες από το glossAPI/Ellinika_Keimena_Project_Gutenberg στο glossAPI/eurlex-greek-legislation.
Για την ασφαλέστερη πρώτη δοκιμή, θα εκτελέσουμε τη Φάση 1 μόνη της ως σύντομο πιλότο και θα προχωρίσουμε στις Φάσεις 2 και 3 μόνο αφού ελέγξουμε τη συμπεριφορά επικύρωσης (validation behavior).

Claude review (to delete once read)
1. The Dictionary Problem Needs More Attention
Correctly identified modern-greek-dictionary as the highest-gap dataset (PPL=21.7) and wisely capped it at 18% and placed it last in the curriculum. But dictionary text has a structural problem that averaging alone won't fix: it trains the model to complete definitions, not to use words in context.
A practical fix: before injecting the dictionary into CPT, run a synthetic transformation pass through GSDG. Convert dictionary entries into contextual sentences:
Input:  "αρμοδιότητα: η εξουσία ή το δικαίωμα να αποφασίζει κάποιος"
Output: Q: Τι σημαίνει αρμοδιότητα σε νομικό πλαίσιο;
        A: Η αρμοδιότητα αναφέρεται στην εξουσία που έχει ένα όργανο...
This transforms the most structurally unusual dataset into running prose — the format the model actually needs.

2. Your Tokenizer Section is Marked WIP — This Blocks Everything Else
The document explicitly sa
"θα πρέπει να γίνει επαναπροσδιορισμός σε σχέση με τα επιλεγμένα datasets"
This is the most urgent dependency. CPT cannot be meaningfully evaluated until the tokenizer is finalized, because:
Perplexity measurements will shift after tokenizer expansion
The training mix weights were calculated on pre-expansion token counts
Fertility improvements change how many tokens each dataset actually contributes to the budget
Suggestion: Before Phase 1 begins, re-run the full perplexity measurement with the expanded tokenizer and update the ranking table. Expect modern-greek-dictionary's PPL to drop significantly once Greek words tokenize as units rather than character sequences — which may change its priority score.






Methodological Improvements
4. The 90/10 Greek/English Split Needs Domain-Aware Anchoring
Your document proposes a flat 10% English anchor from FineWeb-HQ to prevent catastrophic forgetting. This is correct in principle but the Krikri paper showed a more effective approach: periodic "replay" segments rather than a constant mix.
Instead of 10% English in every batch, consider:
95% Greek for most batches
Every N steps, inject a dedicated English + code replay batch
This is especially important during Phase 3 (dictionary data), which is the most structurally alien content
This mirrors what Krikri called "mini replay segments" and better preserves reasoning capabilities because the model sees coherent English blocks rather than English fragments interspersed with Greek.

5. Greek_PhD_Theses_Corpus is Marked NR — It Shouldn't Be Ignored
Your table shows this dataset has:
Priority Score: 1.631 (high — would be HP if measured)
Quality: 0.968 (excellent)
Novelty: 0.663 (strong)
Missing: PPL measurement
This is almost certainly a very strong CPT candidate. Academic thesis Greek is exactly the kind of formal, domain-diverse, high-quality text that builds robust language understanding. The only reason it's NR is a missing measurement, not a quality problem.
Suggestion: Make running PPL on this dataset the immediate next task. It likely belongs in Phase 2 alongside openarchives.gr, possibly displacing the lower-scoring eurlex in the token budget if the gap signal is strong.

6. No Evaluation Benchmark Defined for CPT Phases
Your smoke test uses GreekMMLU as the validation gate, which is correct. But you need benchmark coverage across all three training phases, not just a go/no-go gate at the start:
Phase
Recommended Benchmarks
After Phase 1
GreekMMLU + Belebele Greek (general language understanding)
After Phase 2
+ Legal domain Q/A probe (from eurlex-synthesized pairs)
After Phase 3
+ Vocabulary probe (can the model use dictionary words correctly in context?)
Final
Full Krikri suite: ARC-EL, TruthfulQA-EL, HellaSwag-EL, MMLU-EL

Without phase-level evaluation, you can't distinguish which phase caused a regression if one appears.

7. Annealing Phase is Missing from the Plan
Your CPT plan goes directly from Phase 3 to SFT. Krikri showed that a short annealing pass between CPT and SFT significantly improves performance:
A curated 3.5B token high-quality subset
Fluency-scored using KenLM (within-dataset normalized perplexity)
Supplemented with synthetic Q/A reasoning pairs
This gave Krikri +2.1 points on Greek benchmarks and +0.8 on English — essentially free performance by refining the CPT checkpoint before SFT begins. Given that you already have GSDG generating Q/A pairs, the synthetic component of annealing is already half-built.

10. Plan for Checkpoint Selection Strategy
Your document doesn't specify how you'll select the final CPT checkpoint. The naive approach (last checkpoint) is often wrong. Two better options:
Option A — Validation loss on a held-out Greek probe set: Reserve 1% of each dataset for validation, monitor loss across all checkpoints, pick the minimum. Simple and reliable.
Option B — Best GreekMMLU score across checkpoints: Evaluate every N steps on GreekMMLU, select the checkpoint with best benchmark performance. More expensive but directly optimizes for the metric you care about.
Krikri used a similar approach for DPO checkpoint selection (largest margin between preferred and dispreferred completions on validation). The same logic applies to CPT.

Summary Priority Order
IMMEDIATE (blocks everything):
  1. Finalize tokenizer → re-run perplexity measurements
  2. Measure PPL on Greek_PhD_Theses_Corpus

BEFORE CPT STARTS:
  3. Add reward model filtering to GSDG
  4. Transform dictionary data via GSDG synthesis
  5. Define phase-level evaluation benchmarks
  6. Calculate concrete token budgets for CSCS planning

DURING CPT:
  7. Switch from constant 10% English to periodic replay
  8. Add annealing phase between CPT and SFT

AFTER CPT:
  9. Add multi-turn dialogue generation to GSDG
 10. Define checkpoint selection strategy before SFT begins

Xronopoulos -> Petros Stefaneas



