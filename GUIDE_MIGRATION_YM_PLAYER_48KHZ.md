# Migration de `ym-player` et passage audio à 48 kHz

Ce document décrit la migration réalisée dans `go-dom-intro` depuis la première
version utilisée de `ym-player` jusqu’à la révision actuelle, ainsi que les
adaptations apportées au flux audio Go/Ebitengine et le passage de 44,1 à
48 kHz.

Il couvre également les optimisations de rendu intégrées au même moment, car
elles ont été mesurées conjointement sur desktop et sur un Pixel 10a.

État de référence : 15 septembre 2026.

Implémentation de référence :
[commit `a02358c`](https://github.com/olivierh59500/go-dom-intro/commit/a02358c5470366449f0fadcd25c1ba1f358e44be).

## 1. Versions de départ et d’arrivée

| Élément | Avant | Après |
|---|---|---|
| `ym-player` | `v0.0.0-20250607015657-bb5818debd02` | `v0.0.0-20260913215440-3f73bdca82e5` |
| Ebitengine | `v2.9.11` | `v2.9.11` |
| Fréquence de synthèse | 44 100 Hz | 48 000 Hz |
| Format fourni à Ebitengine | PCM 16 bits stéréo, little-endian | identique |
| Allocation dans `YMPlayer.Read` | variable selon l’étape historique | zéro à l’état final |

Ebitengine reste volontairement en version 2.9.11. La version 2.10.2 a été
compilée et testée, mais elle n’a pas accéléré cette démo et a consommé davantage
de mémoire sur le Pixel pendant les mesures. Une migration Ebitengine doit donc
être traitée séparément d’une migration `ym-player`.

Révision `ym-player` utilisée :

<https://github.com/olivierh59500/ym-player/commit/3f73bdca82e5141870abb65a26b10a3ab8227547>

Fichiers concernés par l’implémentation :

| Fichier | Rôle |
|---|---|
| `go.mod`, `go.sum` | version validée de `ym-player` |
| `game.go` | synthèse, conversion PCM, 48 kHz et rendu partagé |
| `game_test.go` | format stéréo et absence d’allocation audio |
| `cmd/domintro/main.go` | activation du skip-draw desktop |
| `cmd/domintro/draw_on_update.go` | garde contre les rendus redondants |
| `cmd/domintro/draw_on_update_test.go` | tests de la garde desktop |

Le pont `mobile/mobile.go`, l’activité Java et le script Android ne demandent
aucune adaptation spécifique à la nouvelle API `ym-player`.

## 2. Architecture audio retenue

La chaîne finale est la suivante :

```text
assets/eliminator.ym
        │ go:embed
        ▼
stsound.StSound à 48 000 Hz
        │ Compute([]int16)
        ▼
tampon mono réutilisé de 4096 échantillons
        │ volume / 2 + duplication gauche/droite
        ▼
YMPlayer.Read([]byte)
        │ PCM 16 bits stéréo little-endian
        ▼
audio.Player Ebitengine
        ▼
Oto → CoreAudio / AAudio
```

Le paquet `pkg/audio` de `ym-player` n’est pas utilisé ici. Ebitengine possède
déjà son propre contexte audio et son propre cycle de vie multiplateforme. Le
jeu utilise uniquement `pkg/stsound` comme synthétiseur et laisse Ebitengine
gérer la sortie physique.

Cette séparation évite deux contextes Oto concurrents et conserve le même code
de jeu sur desktop et Android.

## 3. Ce qui a changé dans `ym-player`

La migration complète s’est faite en trois états :

| État | Allocations locales de `YMPlayer.Read` | Allocation cachée de `StSound.Compute` |
|---|---:|---:|
| adaptateur initial | au moins 2 par lecture | 1 par bloc |
| premier nettoyage de `go-dom-intro` | 0 | 1 par bloc de 4096 échantillons |
| état actuel | 0 | 0 |

Il fallait donc optimiser à la fois l’adaptateur de l’application et la
bibliothèque. Ne corriger qu’un seul des deux niveaux laissait des allocations
dans la callback audio.

### 3.1 Ancien coût caché de `StSound.Compute`

Dans l’ancienne révision, chaque appel à `Compute` créait un tampon temporaire,
calculait les échantillons, puis les recopiait :

```go
func (s *StSound) Compute(buffer []int16, nbSamples int) bool {
    ymBuffer := make([]YmSample, nbSamples)
    result := s.music.Update(ymBuffer, nbSamples) == YmTrue

    for i := 0; i < nbSamples; i++ {
        buffer[i] = int16(ymBuffer[i])
    }
    return result
}
```

Même si `go-dom-intro` réutilisait son propre tampon, cette allocation interne
restait présente à chaque lecture audio.

### 3.2 `YmSample` est maintenant un alias

La nouvelle version déclare :

```go
type YmSample = int16
```

Un `[]int16` ordinaire peut donc être remis directement au synthétiseur sans
conversion de type ni copie intermédiaire.

### 3.3 Calcul direct dans le tampon appelant

`StSound.Compute` est maintenant réduit à :

```go
func (s *StSound) Compute(buffer []int16, nbSamples int) bool {
    return s.music.Update(buffer, nbSamples) == YmTrue
}
```

Cette modification supprime la dernière allocation cachée du chemin utilisé par
`YMPlayer.Read`.

### 3.4 Autres optimisations internes utiles

La révision actuelle apporte aussi :

- des tables de volume et d’enveloppe immuables, calculées une seule fois ;
- un chemin rapide `updateSimple` pour les morceaux sans effets SID/digidrum ;
- un masque indiquant les voix ayant réellement un effet actif ;
- des accumulateurs conservés dans des variables locales pendant un bloc ;
- moins de trafic mémoire dans la boucle par échantillon ;
- la réutilisation du tableau de volume du tracker ;
- l’emploi de `clear` pour les remises à zéro de tampons ;
- des tests de régression supplémentaires pour le synthétiseur et le
  décompresseur LZH.

Ces changements appartiennent à `ym-player`. Il ne faut pas les recopier dans
`go-dom-intro` : la mise à jour du module suffit.

## 4. Mettre à jour la dépendance Go

### 4.1 Vérifier la version courante

```sh
go list -m github.com/olivierh59500/ym-player
rg 'github.com/olivierh59500/ym-player' go.mod go.sum
```

### 4.2 Installer la révision validée

Depuis la racine de `go-dom-intro` :

```sh
go get \
  github.com/olivierh59500/ym-player@v0.0.0-20260913215440-3f73bdca82e5
go mod tidy
```

Le résultat attendu dans `go.mod` est :

```go
require (
    github.com/hajimehoshi/ebiten/v2 v2.9.11
    github.com/olivierh59500/ym-player v0.0.0-20260913215440-3f73bdca82e5
    golang.org/x/image v0.43.0
)
```

`go.sum` doit également changer. Il doit rester versionné avec `go.mod`.

### 4.3 Ne pas désaligner Ebitengine et `ebitenmobile`

La migration de `ym-player` ne demande pas de changer Ebitengine. Le script
Android continue donc à employer :

```sh
go run github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@v2.9.11
```

La version située après `@` doit toujours correspondre à celle de `go.mod`.

## 5. Adapter `YMPlayer` dans `game.go`

### 5.1 Supprimer l’état inutilisé

L’ancien adaptateur conservait des informations qui n’étaient jamais lues :

```go
type YMPlayer struct {
    player       *stsound.StSound
    sampleRate   int
    buffer       []int16
    mutex        sync.Mutex
    position     int64
    totalSamples int64
    loop         bool
    volume       float64
}
```

La structure finale ne garde que l’état nécessaire au flux :

```go
type YMPlayer struct {
    player *stsound.StSound
    buffer []int16
    mutex  sync.Mutex
    loop   bool
}
```

Conséquences :

- plus d’appel à `GetInfo` uniquement destiné à calculer `totalSamples` ;
- plus d’incrément de `position` à chaque bloc ;
- plus de volume flottant stocké dans la structure ;
- moins d’état à maintenir et à synchroniser.

### 5.2 Construire le synthétiseur avec la fréquence commune

```go
func NewYMPlayer(data []byte, sampleRate int, loop bool) (*YMPlayer, error) {
    player := stsound.CreateWithRate(sampleRate)

    if err := player.LoadMemory(data); err != nil {
        player.Destroy()
        return nil, fmt.Errorf("failed to load YM data: %w", err)
    }

    player.SetLoopMode(loop)

    return &YMPlayer{
        player: player,
        buffer: make([]int16, 4096),
        loop:   loop,
    }, nil
}
```

Le tampon mono de 4096 échantillons est créé une fois, puis réutilisé pendant
toute la lecture.

### 5.3 Écrire directement dans le tampon PCM Ebitengine

Le premier adaptateur de l’application créait deux objets par lecture :

```go
outBuffer := make([]int16, samplesNeeded*2)
buf := make([]byte, 0, len(outBuffer)*2)
```

Il faut éviter ces allocations dans une callback audio. La version finale écrit
directement dans `p` :

```go
func (y *YMPlayer) Read(p []byte) (n int, err error) {
    y.mutex.Lock()
    defer y.mutex.Unlock()

    samplesNeeded := len(p) / 4
    processed := 0

    for processed < samplesNeeded {
        chunkSize := samplesNeeded - processed
        if chunkSize > len(y.buffer) {
            chunkSize = len(y.buffer)
        }

        if !y.player.Compute(y.buffer[:chunkSize], chunkSize) {
            if !y.loop {
                clear(p[processed*4 : samplesNeeded*4])
                err = io.EOF
                break
            }
        }

        for i := 0; i < chunkSize; i++ {
            sample := y.buffer[i] / 2
            offset := (processed + i) * 4

            p[offset] = byte(sample)
            p[offset+1] = byte(sample >> 8)
            p[offset+2] = byte(sample)
            p[offset+3] = byte(sample >> 8)
        }

        processed += chunkSize
    }

    return samplesNeeded * 4, err
}
```

Chaque frame PCM occupe quatre octets :

```text
octet 0 : gauche, poids faible
octet 1 : gauche, poids fort
octet 2 : droite, poids faible
octet 3 : droite, poids fort
```

Le YM est mono. Le même échantillon est donc écrit dans les canaux gauche et
droit.

### 5.4 Remplacer le volume flottant par une division entière

Le volume historique était fixé à `0.5` :

```go
sample := int16(float64(y.buffer[i]) * 0.5)
```

La forme suivante produit le même arrondi vers zéro pour un entier signé, sans
conversion flottante dans la boucle audio :

```go
sample := y.buffer[i] / 2
```

Si le volume doit devenir réglable, utiliser plutôt `audio.Player.SetVolume` ou
un facteur fixe entier. Ne pas réintroduire une allocation de tampon pour cette
fonctionnalité.

### 5.5 Conserver le verrou autour de `Read` et `Close`

Le mutex empêche une destruction du synthétiseur pendant une lecture en cours.
Son coût est faible face au risque d’accès concurrent à `StSound`.

La fermeture est idempotente :

```go
func (y *YMPlayer) Close() error {
    y.mutex.Lock()
    defer y.mutex.Unlock()

    if y.player != nil {
        y.player.Destroy()
        y.player = nil
    }
    return nil
}
```

## 6. Passer correctement à 48 kHz

### 6.1 Pourquoi 48 kHz sur Android

Le Pixel testé annonce une sortie matérielle à 48 000 Hz :

```sh
adb shell dumpsys media.audio_flinger | \
  grep -Ei 'sample rate|sampling rate'
```

Résultat observé :

```text
Sample rate: 48000 Hz
```

À 44,1 kHz, AAudio/Oboe insérait un rééchantillonneur. Le profil CPU contenait
notamment :

```text
oboe::flowgraph::SampleRateConverter::onProcess
oboe::resampler::PolyphaseResamplerStereo::readFrame
```

Le passage à 48 kHz évite le changement de fréquence et réduit ce travail sur le
Pixel. Il produit 8,84 % d’échantillons YM supplémentaires, mais le synthétiseur
actuel est assez rapide pour que le bilan mesuré reste positif. Dans le prototype
A/B, les cycles, les instructions et le temps CPU total ont diminué d’environ
2 % ; il faut considérer cela comme un petit gain, pas comme l’optimisation
principale.

### 6.2 Une seule constante pour les deux côtés

Modifier :

```go
const sampleRate = 44100
```

en :

```go
const sampleRate = 48000
```

Cette même constante doit impérativement être utilisée par :

```go
g.audioContext = audio.NewContext(sampleRate)
ym, err := NewYMPlayer(ymData, sampleRate, true)
```

Il ne faut pas créer le synthétiseur à 44,1 kHz et annoncer 48 kHz à Ebitengine,
ou inversement : cela modifierait la hauteur et la durée du morceau.

### 6.3 Initialiser l’audio au premier `Update`

Sur Android, `mobile.SetGame` construit le jeu pendant le chargement de la
bibliothèque native, avant que l’activité soit entièrement prête. Le contexte
audio reste donc différé :

```go
func (g *Game) Update() error {
    if !g.audioReady {
        g.audioReady = true
        g.initAudio()
        g.startMusic()
    }

    // Mise à jour du jeu...
    return nil
}
```

Ne pas remettre `initAudio` dans `NewGame`. Cela avait provoqué un écran blanc
sur Android lors d’une adaptation précédente.

### 6.4 Validation auditive

Après installation, contrôler :

- la hauteur du morceau ;
- sa vitesse ;
- l’absence de crépitement ;
- l’absence de coupure après mise en veille/reprise ;
- le bouclage en fin de morceau.

Le test automatisé vérifie le format PCM et les allocations, mais une écoute
reste nécessaire pour valider le timbre.

## 7. Optimisations de rendu intégrées en même temps

Ces changements ne sont pas requis par la nouvelle version de `ym-player`, mais
ils font partie de l’état actuel de `go-dom-intro`.

### 7.1 Calculer uniquement le scrolling actif

Auparavant, les quatre tailles de texte étaient effacées et redessinées à chaque
tick. La version actuelle :

1. avance un offset de référence ;
2. synchronise les offsets des autres tailles ;
3. détermine `actSize` ;
4. dessine uniquement le `ScrollText` actif.

Les quatre canvases restent alloués pour éviter une allocation ou un upload lors
d’un changement de taille.

### 7.2 Mettre en cache les sous-images stables

Les frames d’étoiles, les tranches paires du raster de fond et la bande
supérieure de `mergeCanvas` sont créées une seule fois par `cacheSubImages`.

Cela évite de reconstruire de nombreux objets `SubImage` dans `Draw`.

### 7.3 Dessiner directement les images complètes

L’ancien helper générique `drawPart` recalculait systématiquement un rectangle,
une sous-image et deux boucles de répétition. Il a été remplacé par deux helpers
spécialisés :

```go
drawImageAt(dest, src, x, y)
drawRepeatedVertically(dest, src, y, step, count)
```

Ebitengine peut ainsi regrouper plus simplement les dessins utilisant la même
texture et le même mode de fusion.

### 7.4 Regrouper les cibles de rendu

L’ordre final est :

```text
scrollCanvas actif → offScroll → mergeCanvas → écran
```

Les passes hors écran sont préparées avant le fond principal. Cela évite des
allers-retours inutiles entre l’écran et les canvases intermédiaires.

### 7.5 Employer l’API de fusion actuelle

Remplacer :

```go
op.CompositeMode = ebiten.CompositeModeDestinationIn
```

par :

```go
op.Blend = ebiten.BlendDestinationIn
```

Le résultat Porter-Duff est identique, sans utiliser l’ancien champ déprécié.

### 7.6 Ne pas redessiner une scène desktop inchangée

Sur le Mac de test, l’écran tourne à environ 165 Hz, tandis que le jeu utilise
60 `Update` par seconde. Sans garde, `Draw` reconstruit donc plusieurs fois le
même état.

Le lanceur desktop appelle :

```go
ebiten.SetScreenClearedEveryFrame(false)
ebiten.RunGame(newDrawOnUpdateGame(domintro.NewGame()))
```

`drawOnUpdateGame` ne transmet qu’un seul `Draw` après chaque `Update`. Cette
optimisation reste dans `cmd/domintro` et n’affecte pas le pont Android.

## 8. Tests ajoutés

### 8.1 Flux YM

`TestYMPlayerReadProducesStereoWithoutAllocating` vérifie :

- le chargement de `eliminator.ym` ;
- le nombre d’octets retourné ;
- l’identité des canaux gauche et droit ;
- zéro allocation par appel à `Read` avec `testing.AllocsPerRun`.

### 8.2 Texte

`TestTextTilesStripsControlCodesAndPreservesSpacing` protège la conversion du
texte et le retrait des codes `^CsN;`.

### 8.3 Skip-draw desktop

Les tests du lanceur vérifient :

- un rendu initial unique ;
- aucun rendu répété sans nouvel `Update` ;
- un rendu après une mise à jour réussie ;
- aucune nouvelle image après un `Update` en erreur.

### 8.4 Commandes de validation

```sh
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

La dernière commande suppose que `golangci-lint` est installé.

Sur macOS, un vrai test d’un paquet important Ebitengine peut demander l’accès à
la session graphique. Exécuter ces commandes depuis un Terminal de la session
utilisateur, ou prévoir un environnement graphique adapté en CI.

## 9. Reproduire le benchmark du synthétiseur

Pour reproduire le benchmark, créer temporairement
`ym_compute_benchmark_test.go` à la racine du projet :

```go
package domintro

import (
    "os"
    "testing"

    "github.com/olivierh59500/ym-player/pkg/stsound"
)

func BenchmarkCompute4096(b *testing.B) {
    data, err := os.ReadFile("assets/eliminator.ym")
    if err != nil {
        b.Fatal(err)
    }

    player := stsound.CreateWithRate(44100)
    defer player.Destroy()
    if err := player.LoadMemory(data); err != nil {
        b.Fatal(err)
    }
    player.SetLoopMode(true)

    buffer := make([]int16, 4096)
    b.ReportAllocs()
    b.ResetTimer()
    for b.Loop() {
        if !player.Compute(buffer, len(buffer)) {
            b.Fatal("unexpected end of looping stream")
        }
    }
}
```

Commande :

```sh
go test -run '^$' \
  -bench BenchmarkCompute4096 \
  -benchmem \
  -benchtime=1s \
  -count=5
```

Résultats moyens observés sur Apple M4 Max, avec un bloc de 4096 échantillons :

| Version `ym-player` | Temps | Octets/op | Allocations/op |
|---|---:|---:|---:|
| juin 2025 | 30,4 µs | 8192 | 1 |
| septembre 2026 | 8,42 µs | 0 | 0 |

Le synthétiseur est environ 3,6 fois plus rapide sur ce morceau. Le gain CPU de
l’application complète est plus faible, car la présentation Android et le GPU
restent les principaux consommateurs.

Ce micro-benchmark comparatif utilise 44,1 kHz afin de reproduire exactement la
configuration de départ. Le nombre d’échantillons par bloc reste identique ; la
configuration finale à 48 kHz est validée séparément dans l’application complète.

## 10. Construire et tester sur Android

Le Pixel étant branché, déverrouillé et autorisé :

```sh
./scripts/run-android.sh
```

Le script :

1. génère `android/app/libs/domintro.aar` pour `arm64-v8a` ;
2. compile l’APK debug ;
3. vérifie qu’un seul appareil est disponible ;
4. installe l’APK avec `adb install -r` ;
5. force l’arrêt de l’ancienne instance ;
6. lance `com.olivierh.domintro/.MainActivity`.

### Contrôler les erreurs audio

```sh
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb

"$ADB" logcat -c
"$ADB" shell am force-stop com.olivierh.domintro
"$ADB" shell am start -n com.olivierh.domintro/.MainActivity
sleep 4
"$ADB" logcat -d -v threadtime | \
  grep -Ei 'FATAL EXCEPTION|panic|GoLog|underrun|AAudio|Oboe'
```

### Contrôler l’activité au premier plan

```sh
"$ADB" shell pidof com.olivierh.domintro
"$ADB" shell dumpsys window displays | \
  grep 'mCurrentFocus=.*com.olivierh.domintro'
```

### Contrôler l’APK

```sh
AAPT="$ANDROID_HOME/build-tools/36.0.0/aapt"
APKSIGNER="$ANDROID_HOME/build-tools/36.0.0/apksigner"
ZIPALIGN="$ANDROID_HOME/build-tools/36.0.0/zipalign"
APK=android/app/build/outputs/apk/debug/app-debug.apk

"$AAPT" dump badging "$APK"
"$APKSIGNER" verify --verbose "$APK"
"$ZIPALIGN" -c -P 16 -v 4 "$APK"
```

## 11. Résultats mesurés dans l’application complète

### Pixel 10a

Comparaison à Ebitengine identique, avant et après l’ensemble des optimisations :

| Mesure | Avant | Après | Évolution |
|---|---:|---:|---:|
| FPS présentés | 60,05 | 60,06 | plafond conservé |
| VSync corrects | 62/62 | 62/62 | aucun raté |
| Instructions CPU | 1,069 milliard | 0,971 milliard | −9,1 % |
| Cycles CPU | 2,697 milliards | 2,598 milliards | −3,7 % |
| Temps CPU sur 5 s | 3,681 s | 3,588 s | −2,5 % |
| RSS | 364,6 Mo | 365,5 Mo | +0,9 Mo |

Ces résultats incluent la migration audio et les optimisations de rendu de la
section 7. Ils ne doivent pas être attribués uniquement au changement de
fréquence.

### Desktop à environ 165 Hz

| Mesure sur 10 s | Avant | Après | Évolution |
|---|---:|---:|---:|
| Allocations cumulées | 101,9 Mo | 56,8 Mo | −44 % |
| Objets alloués | 2,94 millions | 1,35 million | −54 % |
| Cycles de GC | 19 | 11 | −42 % |
| Instructions | 7,88 milliards | 5,66 milliards | −28 % |
| Cycles CPU | 5,88 milliards | 4,89 milliards | −17 % |

Le gain desktop provient principalement du skip-draw sur l’écran à haute
fréquence. Le jeu continue à évoluer à 60 TPS.

## 12. Produire des binaires de distribution plus petits

Les builds debug doivent conserver leurs symboles pour le profilage. Pour une
distribution, générer la bibliothèque mobile avec :

```sh
-trimpath -ldflags='-s -w'
```

Exemple complet :

```sh
go run github.com/hajimehoshi/ebiten/v2/cmd/ebitenmobile@v2.9.11 \
  bind \
  -target android/arm64 \
  -androidapi 23 \
  -javapkg com.olivierh.domintro \
  -trimpath \
  -ldflags='-s -w' \
  -o android/app/libs/domintro.aar \
  ./mobile
```

Mesures observées :

| Artefact | Normal | Dépouillé |
|---|---:|---:|
| exécutable desktop | 11 Mo | 6,9 Mo |
| AAR Android | 7,2 Mo | 2,8 Mo |
| `libgojni.so` | 13,35 Mo | 7,67 Mo |

Il est préférable de créer un script de release distinct plutôt que de retirer
les symboles du script `run-android.sh` utilisé pour le diagnostic.

## 13. Retour arrière

### Revenir uniquement à 44,1 kHz

Dans `game.go` :

```go
const sampleRate = 44100
```

Comme le contexte Ebitengine et le synthétiseur partagent cette constante, aucune
autre ligne n’est nécessaire.

### Revenir à l’ancienne dépendance

```sh
go get \
  github.com/olivierh59500/ym-player@v0.0.0-20250607015657-bb5818debd02
go mod tidy
go test ./...
```

Attention : cette version réintroduit une allocation de 8192 octets dans
`StSound.Compute` pour chaque bloc de 4096 échantillons.

### Restaurer l’APK après un essai

```sh
./scripts/run-android.sh
```

Cette commande reconstruit et réinstalle l’état exact du dépôt courant.

## 14. Diagnostic rapide

### Son trop aigu ou trop lent

Vérifier que la même valeur est utilisée dans :

- `audio.NewContext(sampleRate)` ;
- `stsound.CreateWithRate(sampleRate)`.

### Crépitements ou sous-alimentation

- rechercher `underrun`, `AAudio` et `Oboe` dans `adb logcat` ;
- vérifier que `YMPlayer.Read` n’alloue pas ;
- ne pas réduire arbitrairement le tampon réutilisé ;
- vérifier que l’application tient toujours 60 FPS ;
- contrôler les économies d’énergie ou limitations thermiques du téléphone.

### Écran blanc au démarrage Android

Vérifier que l’audio est initialisé au premier `Update` et non dans `NewGame`.

### APK anormalement gros après plusieurs essais d’AAR

Gradle peut conserver des données orphelines dans un APK incrémental après des
remplacements successifs de bibliothèques natives. Reconstruire proprement :

```sh
./android/gradlew -p android clean assembleDebug
```

Lors des essais réalisés ici, cette opération a ramené un APK temporairement
gonflé de 25 Mo à sa taille réelle de 13 Mo.

## 15. Checklist de migration

- [ ] Relever la version `ym-player` de départ.
- [ ] Mesurer le comportement avant modification.
- [ ] Mettre à jour vers `v0.0.0-20260913215440-3f73bdca82e5`.
- [ ] Exécuter `go mod tidy`.
- [ ] Conserver Ebitengine et `ebitenmobile` à la même version.
- [ ] Réutiliser un tampon mono `[]int16`.
- [ ] Écrire directement le PCM stéréo dans le tampon Ebitengine.
- [ ] Éviter les conversions flottantes pour un volume constant de 50 %.
- [ ] Utiliser 48 000 Hz des deux côtés de la chaîne.
- [ ] Initialiser l’audio au premier `Update` sur Android.
- [ ] Vérifier zéro allocation dans `YMPlayer.Read`.
- [ ] Exécuter les tests, le détecteur de courses, `vet` et le lint.
- [ ] Construire et installer l’APK sur l’appareil réel.
- [ ] Vérifier logs, pause/reprise, bouclage et qualité à l’écoute.
- [ ] Conserver les symboles en debug et les retirer uniquement en release.
