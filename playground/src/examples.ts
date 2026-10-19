// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type { Schema } from "./api";

export type Example = { name: string; schema: Schema; text: string };

export const EXAMPLES: Example[] = [
  {
    // Sentences from English Wikipedia (CC BY-SA 4.0), via the
    // knowledge-graphs-workshop ASoIaF corpus.
    name: "ASoIaF",
    schema: {
      entities: ["character", "house", "castle", "kingdom"],
      relations: ["child of", "sibling of", "married to", "lord of", "member of"],
    },
    text: [
      "Robb Stark is the eldest child of Eddard and Catelyn Stark and the heir to Winterfell.",
      "Catelyn is usually in accord with her husband Eddard, but resents his admission of his extramarital son, Jon Snow, into their household.",
      "Jon confides he is the son of Rhaegar Targaryen and Lyanna Stark after swearing Sansa and Arya to secrecy.",
      "Sansa wakes one morning to the shock of being fitted for a wedding gown—she is to wed Tyrion that same day in a forced marriage.",
      "Tywin is the ruthless patriarch of House Lannister of Casterly Rock and father to twins Cersei and Jaime, and Tyrion.",
      "Tywin Lannister is Lord of Casterly Rock, Shield of Lannisport, and Warden of the West.",
      "Jaime Lannister is the twin brother of Cersei and the first son of Tywin Lannister.",
      "Robert later married Cersei Lannister to secure House Lannister's support for his rule.",
      "Introduced in 1996's A Game of Thrones, Tommen Baratheon is the youngest son of Robert Baratheon and Cersei Lannister from the kingdom of Westeros.",
    ].join("\n"),
  },
  {
    name: "Orchard Labs",
    schema: {
      entities: ["person", "organization", "location", "food"],
      relations: ["works_at", "works_with", "founded", "located_in", "likes", "studied_at"],
    },
    text: [
      "Johnny likes apples.",
      "Mindy works with Johnny at Orchard Labs.",
      "Priya Raman founded Orchard Labs in Portland.",
      "Orchard Labs grows heirloom apples in Hood River.",
      "Mindy studied botany at Oregon State University.",
      "Priya hired Mindy after a talk at Oregon State University.",
      "Johnny grew up in Hood River and still likes pears more than apples.",
    ].join("\n"),
  },
  {
    name: "Film",
    schema: {
      entities: ["person", "film", "studio", "award"],
      relations: ["directed", "acted_in", "produced_by", "won"],
    },
    text: [
      "Greta Gerwig directed Little Women.",
      "Saoirse Ronan acted in Little Women and Lady Bird.",
      "Greta Gerwig directed Lady Bird for A24.",
      "Lady Bird was produced by A24.",
      "Timothée Chalamet acted in Lady Bird and Dune.",
      "Dune won the Academy Award for Best Cinematography.",
    ].join("\n"),
  },
  {
    name: "Startups",
    schema: {
      entities: ["person", "company", "product", "city"],
      relations: ["founded", "acquired", "makes", "headquartered_in", "works_at"],
    },
    text: [
      "Ada Okafor founded Lumen Robotics in Lagos.",
      "Lumen Robotics makes the Firefly delivery drone.",
      "Northwind acquired Lumen Robotics in 2024.",
      "Northwind is headquartered in Seattle.",
      "Ada Okafor now works at Northwind.",
      "Northwind makes the Atlas warehouse robot.",
    ].join("\n"),
  },
];
