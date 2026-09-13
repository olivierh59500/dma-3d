# DMA 3D

Démo Glenz vectors écrite en Go avec Ebitengine et `ym-player`.

## Version ordinateur

```sh
go run ./cmd/dma3d
```

## Pixel / Android

Avec un unique appareil Android autorisé et connecté en USB :

```sh
./scripts/run-android.sh
```

Le script génère l’AAR Ebitengine pour `arm64-v8a`, construit l’APK de
débogage, l’installe puis lance `com.olivierh.dma3d/.MainActivity`.

Le guide détaillé de configuration et de diagnostic se trouve dans
[`GUIDE_ANDROID_EBITENGINE_PIXEL.md`](GUIDE_ANDROID_EBITENGINE_PIXEL.md).

## Vérifications Go

```sh
go test ./...
go vet ./...
```
