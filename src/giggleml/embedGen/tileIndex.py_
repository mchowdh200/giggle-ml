import os
import pickle
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, overload, override

import numpy as np

from giggleml.dataWrangling.fasta import ensureFa
from giggleml.dataWrangling.intervalDataset import IntervalDataset, MemoryIntervalDataset
from giggleml.embedGen import embedIO
from giggleml.embedGen.embedIO import Embed
from giggleml.embedGen.embedPipeline import DirectPipeline, EmbedPipeline
from giggleml.intervalTransformer import IntervalTransformer
from giggleml.intervalTransforms import IntervalTransform, Tiling
from giggleml.utils.types import MmapF32


@dataclass
class TileIndexFooter:
    shape: dict[str, tuple[int, tuple[int, int]]]
    tileSize: int
    modelInfo: str | None


class TileIndex(EmbedPipeline):
    def __init__(self, path: str):
        self.masterPath: str = path
        self.maps: dict[str, list[MmapF32]]
        self.eDim: int
        self.tiler: Tiling
        self.modelInfo: str | None
        self.eDim = 128
        # self.load()

    @override
    def __repr__(self) -> str:
        return f"TileIndex({self.masterPath})"

    @classmethod
    def build(
        cls, tileSize: int, fastaPath: str, outPath: str, basePipeline: EmbedPipeline
    ) -> Self:
        """
        Build a new TileIndex by embeddings all possible tiles corresponding to a fasta file.
        Because embedding generation is required, an existing EmbedPipeline is necessary, probably
        a DirectPipeline.

        @param outPath: standard extension .tile
        """
        if isinstance(basePipeline, DirectPipeline):
            if basePipeline.model.maxSeqLen is not None:
                assert tileSize <= basePipeline.model.maxSeqLen

        # truncate out file
        with open(outPath, "w"):
            pass

        shapeMap = dict()
        fasta = ensureFa(fastaPath)
        lastBytes = 0

        for chrm, content in fasta:
            with tempfile.NamedTemporaryFile() as tmpFile:
                tmpPath = tmpFile.name
                assert os.path.isfile(tmpPath)

                print(f"Embedding {chrm}")
                interval = (chrm, 0, len(content))
                embed = basePipeline.embed(MemoryIntervalDataset([interval], fastaPath), tmpPath)

                shapeMap[chrm] = (lastBytes, embed.data.shape)
                embed.unload()

                with open(outPath, "a+b") as master:
                    while True:
                        chunk = tmpFile.read(4096)

                        if not chunk:
                            break

                        master.write(chunk)
                        lastBytes += len(chunk)

                # because the associated metadata file needs to be deleted as well
                embed.delete()

        modelStr = str(basePipeline.model) if isinstance(basePipeline, DirectPipeline) else None
        footer = TileIndexFooter(shapeMap, tileSize, modelStr)
        footerBytes = b"\n" + pickle.dumps(footer)

        with open(outPath, "a+b") as master:
            master.write(footerBytes)

        return cls(outPath)

    def _parseFooter(self) -> TileIndexFooter:
        # const
        fileSize = os.path.getsize(self.masterPath)
        chunkSize = 4096

        # variable
        buffer = b""
        offset = fileSize

        with open(self.masterPath, "rb") as f:
            # Ensure we don't go beyond the beginning of the file
            while offset >= 0:
                f.seek(offset)
                readSize = min(chunkSize, fileSize - offset)  # Don't read beyond the file
                chunk = f.read(readSize)
                buffer = chunk + buffer  # Prepend to existing buffer (reading backwards)
                footerStartByte = buffer.rfind(b"\n")

                if footerStartByte != -1:
                    footerBytes = buffer[footerStartByte + 1 :]
                    footerObj = pickle.loads(footerBytes)
                    hiddenDims = [shape[1] for _, shape in footerObj.values()]

                    if not all([hiddenDim == hiddenDims[0] for hiddenDim in hiddenDims]):
                        raise MisformattedTileIndex(self.masterPath)

                    return footerObj

                offset -= chunkSize

        raise MisformattedTileIndex(self.masterPath)

    def load(self):
        footer = self._parseFooter()
        self.maps = {
            chrm: np.memmap(self.masterPath, np.float32, "r", offset, shape)
            for chrm, (offset, shape) in footer.shape.items()
        }
        self.eDim = next(iter(self.maps.values())).shape[1]
        self.tileSize = footer.tileSize
        self.modelInfo = footer.modelInfo

    @overload
    def embed(
        self,
        intervals: IntervalDataset,
        out: str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Embed: ...

    @overload
    def embed(
        self,
        intervals: Sequence[IntervalDataset],
        out: Sequence[str],
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed]: ...

    @override
    def embed(
        self,
        intervals: Sequence[IntervalDataset] | IntervalDataset,
        out: Sequence[str] | str,
        transforms: list[IntervalTransform] | None = None,
    ) -> Sequence[Embed] | Embed:

        # FIXME: rm transforms parameter

        # FIXME: /.*return/

        if isinstance(intervals, Sequence) == isinstance(out, str):
            raise ValueError("Expecting either both or neither of data & out to be sequences")
        if not isinstance(intervals, Sequence):
            intervals = [intervals]
        if isinstance(out, str):
            out = [out]
        if transforms is None:
            transforms = list()

        # FIXME: rm
        def make():
            size = int(125e6 // 500 * 2)
            files = list()
            for i in range(5):
                file = np.memmap(f"{i}.tmp.npy", np.float32, "w+", shape=(size, self.eDim))
                file[:] = 0
                files.append(file)
            return files

        self.maps = defaultdict(make)

        self.tiler = Tiling(500, 5)
        meta = embedIO.EmbedMeta(self.eDim, np.float32, str(self))
        embeds = list()

        for i, (bed, outPath) in enumerate(zip(intervals, out)):
            outFile: MmapF32 = np.memmap(outPath, np.float32, "w+", shape=(len(bed), self.eDim))

            for j, interval in enumerate(bed):
                tiles = self.tiler.tile(interval)
                chrm = interval[0]
                maps = self.maps[chrm]

                embedChunks = list()

                for layerIdx, layerTiles in enumerate(tiles):
                    tilingLayer = maps[layerIdx]

                    for tileIdx in layerTiles:
                        tile = tilingLayer[tileIdx]
                        embedChunks.append(tile)

                # FIXME: this call is awkward
                weights = self.tiler.weights([interval])
                embedChunks = np.array(embedChunks)
                embedChunks *= weights[:, None]  # pyright: ignore[reportAssignmentType]
                embed = np.mean(embedChunks, axis=0)
                outFile[j] = embed

            embeds.append(embedIO.writeMeta(outFile, meta))
            print(f"{i+1}/{len(intervals)}")
        return embeds


class MisformattedTileIndex(Exception):
    def __init__(self, path: str):
        super().__init__(f"Misformatted TileIndex file, {path}")
