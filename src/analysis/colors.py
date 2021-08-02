from typing import Dict, List, Optional

class ColorWheel:
    def __init__(self, size: Optional[int] = None):
        self.size = size

        self._map: Dict[str, str] = {}
        self._state: int = 0

    def lock(self, labels: List[str]):
        self.size = len(labels)

        for label in labels:
            self._map[label] = self._pickColor()

        return self

    def get(self, label: str):
        if label in self._map:
            return self._map[label]

        color = self._pickColor()
        self._map[label] = color

        return color

    def _pickColor(self):
        # default to the largest palette size if we don't know
        size = self.size if self.size else 6
        palette = self.colorClasses(size)
        color = palette[self._state]

        self._state = (self._state + 1) % size

        return color

    @classmethod
    def colorClasses(cls, size: int):
        classes = {
            3: [ '#1b9e77', '#d95f02', '#7570b3' ],
            4: [ '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c' ],
            # not colorblind safe
            # TODO: find a 5+ class colorblind safe palette
            # 5: [ '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00' ],
            # 6: [ '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33' ],

            # colorblind safe
            5: ['#88CCEE', '#44AA99', '#882255', '#CC6677', '#BBBBBB'],
            6: ['#88CCEE', '#44AA99', '#882255', '#CC6677', '#999933', '#BBBBBB'],
            7: ['#88CCEE', '#44AA99', '#882255', '#CC6677', '#999933', '#000000', '#BBBBBB'],
            8: ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933', '#CC6677', '#882255'],
            9: ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933', '#CC6677', '#882255', '#AA4499'],
            10: ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933', '#CC6677', '#882255', '#AA4499', '#BBBBBB'],
        }

        return classes[size]

basicControlColors = ColorWheel().lock(['DQN', 'ESARSA', 'EQRC'])
