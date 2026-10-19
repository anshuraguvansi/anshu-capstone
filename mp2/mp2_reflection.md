# MP2 Reflection

## What worked

Both retrieval approaches found the expected source for all five questions. In the latest saved runs, hybrid retrieval matched 16/23 fact strings compared with 8/23 for dense retrieval. It identified John Clay as Vincent Spaulding's real identity and correctly explained that the hydraulic press stamped counterfeit half-crowns. I also reused the reranker across questions, included retrieval in the latency measurement, and made interactive and validation runs use three final chunks.

## What didn't work

Dense retrieval found the right story but missed key evidence: it could not identify John Clay and described the press's use as a trap instead of its counterfeiting purpose. Hybrid retrieval improved these answers, but still omitted the swamp adder's name and the photograph's exact hiding place. Its Peterson answer also ambiguously attached the goose's identifying card to both recovered objects. Correct source hits and higher fact scores did not guarantee complete, precise answers.

## What I'd change

I would try chunking by paragraph boundaries and section titles instead of fixed 500-character windows. I would compare the same five questions and read the generated answers alongside the fact scores to check whether related evidence stays together.

## One surprise

The latest dense answer to the Irene Adler question matched 0/5 fact strings even though it explained that she recognised Holmes and left with the photograph. The hybrid answer matched 2/5 while describing the smoke-rocket plan and her departure with her husband. Seeing the actual answers showed me why literal fact matching needs a manual check.

Validation results:

| Run | Q1: Assistant | Q2: Snake | Q3: Peterson | Q4: Press | Q5: Photograph | Total facts | Source hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Latest dense retrieval | 1/5 | 2/6 | 4/4 | 1/3 | 0/5 | 8/23 | 5/5 |
| Latest hybrid + reranker: retrieval k=5, final k=3 | 3/5 | 4/6 | 4/4 | 3/3 | 2/5 | 16/23 | 5/5 |

### Dense retrieval — latest saved run

```text
Validating 2 questions from predefined_questions.jsonl…

  ✓ q1_red_headed_league_assistant
      Q: Who was the assistant working at Jabez Wilson's pawnshop, and what was his real identity?
      Answer: The assistant working at Jabez Wilson's pawnshop was Vincent Spaulding. His real identity, however, is not explicitly provided in the excerpts. The excerpts discuss Spaulding's characteristics and duties but do not reveal his true identity. 

(Source: The Red-Headed League — Section 4 and Section 6)
      Cited: 01_red_headed_league.txt
      Expected: 01_red_headed_league.txt
      Facts matched: 1/5
      Latency: 3910ms

  ✓ q2_speckled_band_murder_weapon
      Q: What killed Julia Stoner at Stoke Moran, and how was the method discovered?
      Answer: Julia Stoner was killed by a venomous snake that Doctor Roylott had trained to kill her. The method was discovered when Sherlock Holmes noticed that a small vent was leading into Julia’s room, through which the snake could have come. Holmes also deduced that the whistle that Julia had heard was a signal to the snake to return. This information, combined with the prior evidence of peculiar circumstances surrounding Julia's death, led to the conclusion of how she was murdered. 

(Source: The Adventure of the Speckled Band — Section 3)
      Cited: 02_speckled_band.txt
      Expected: 02_speckled_band.txt
      Facts matched: 2/6
      Latency: 3092ms

  Source-match: 2/2

  Validating 3 questions from learner_questions.jsonl…

  ✓ q3_blue_carbuncle_hat_and_goose
      Q: What did Peterson witness on Christmas morning in Tottenham Court Road, and what did he do afterward?
      Answer: On Christmas morning, Peterson witnessed a tall man in a black coat carrying a goose being accosted by a group of roughs in Tottenham Court Road. During the scuffle, the man's hat fell off, and as Peterson rushed to assist, the man dropped the goose and fled. Afterward, Peterson rescued both the goose and the hat, and he took them to Sherlock Holmes for further inquiries (The Adventure of the Blue Carbuncle — Sections 1 and 2).
      Cited: 03_blue_carbuncle.txt
      Expected: 03_blue_carbuncle.txt
      Facts matched: 4/4
      Latency: 2806ms

  ✓ q4_engineers_thumb_hydraulic_press
      Q: What did Colonel Stark claim the hydraulic press was used for, and what was its actual purpose?
      Answer: Colonel Stark claimed that the hydraulic press was used to compress fuller's earth. However, its actual purpose was a trap designed to crush an unwanted visitor, as Hatherley discovered when he realized the chamber was a kill mechanism. The evidence showed that the press had never been used for its claimed purpose, as the floorwas marked by metallic deposits rather than any clay-like substance. 

(Source: The Adventure of the Engineer's Thumb — Sections 8 and 9)
      Cited: 04_engineers_thumb.txt
      Expected: 04_engineers_thumb.txt
      Facts matched: 1/3
      Latency: 2616ms

  ✓ q5_scandal_in_bohemia_photograph_and_escape
      Q: How did Holmes discover where Irene Adler hid the photograph, and why did he fail to recover it the next morning?
      Answer: Holmes discovered where Irene Adler hid the photograph by observing her actions on the evening when he disguised himself as a clergyman. He noted that she placed great value on the photograph, predicting that she would secure it in a way that was meaningful to her. When a diversion occurred outside her home, he was able to watch where she ran, leading him to the hidden location.

However, he failed to recover the photograph the next morning because Irene Adler had recognized him during his disguise. She anticipated his plan and left the country with the photograph, leaving behind a letter for Holmes explaining her departure. The letter stated that she had recognized him and decided to take the photograph with her, thus outsmarting Holmes. 

(Source: A Scandal in Bohemia — Sections 11, 13, and 14)
      Cited: 05_scandal_in_bohemia.txt
      Expected: 05_scandal_in_bohemia.txt
      Facts matched: 0/5
      Latency: 3149ms

  Source-match: 3/3
```

### Hybrid retrieval — latest saved run

```text
   Validating 2 questions from predefined_questions.jsonl…

  ✓ q1_red_headed_league_assistant
      Q: Who was the assistant working at Jabez Wilson's pawnshop, and what was his real identity?
      Answer: The assistant working at Jabez Wilson's pawnshop was Vincent Spaulding, who was, in reality, John Clay, the grandson of a royal duke and a notorious thief. (Source: The Red-Headed League — Section 4 and Section 8)
      Cited: 01_red_headed_league.txt
      Expected: 01_red_headed_league.txt
      Facts matched: 3/5
      Latency: 4105ms

  ✓ q2_speckled_band_murder_weapon
      Q: What killed Julia Stoner at Stoke Moran, and how was the method discovered?
      Answer: Julia Stoner was killed by a snake, specifically a "speckled band," which was trained and used by her stepfather, Doctor Roylott. The method of her death wasdiscovered when Holmes and Watson examined Julia's old room, where Holmes found a dummy bell-rope and a ventilator that connected to Doctor Roylott's room, allowing the snake to slither through to Julia's room. This connection and the presence of the bell-rope were key in unraveling the mystery of her death.

(Source: The Adventure of the Speckled Band — Sections 1, 8, and 9)
      Cited: 02_speckled_band.txt
      Expected: 02_speckled_band.txt
      Facts matched: 4/6
      Latency: 3170ms

  Source-match: 2/2

  Validating 3 questions from learner_questions.jsonl…

  ✓ q3_blue_carbuncle_hat_and_goose
      Q: What did Peterson witness on Christmas morning in Tottenham Court Road, and what did he do afterward?
      Answer: On Christmas morning, Peterson witnessed a tall man in a black coat being accosted by a group of roughs while he was carrying a fat goose over his shoulder. During the scuffle, the man's hat fell off, and as Peterson rushed to assist, the man dropped the goose and fled. After the incident, Peterson rescued the goose and the hat, which had a card attached marked "For Mrs Henry Baker." He took both objects to Sherlock Holmes, who agreed to make inquiries about them. The goose was later sent home with Peterson, who was advised to eat it before it spoiled (The Adventure of the Blue Carbuncle — Section 2).
      Cited: 03_blue_carbuncle.txt
      Expected: 03_blue_carbuncle.txt
      Facts matched: 4/4
      Latency: 3291ms

  ✓ q4_engineers_thumb_hydraulic_press
      Q: What did Colonel Stark claim the hydraulic press was used for, and what was its actual purpose?
      Answer: Colonel Stark claimed that the hydraulic press was used to compress fuller's earth into bricks. However, its actual purpose was to stamp counterfeit half-crowns from a base alloy of nickel and tin. This was confirmed when Holmes deduced that the press was involved in a counterfeiting operation. 

(Source: The Adventure of the Engineer's Thumb — Section 4 and Section 13)
      Cited: 04_engineers_thumb.txt
      Expected: 04_engineers_thumb.txt
      Facts matched: 3/3
      Latency: 2713ms

  ✓ q5_scandal_in_bohemia_photograph_and_escape
      Q: How did Holmes discover where Irene Adler hid the photograph, and why did he fail to recover it the next morning?
      Answer: Holmes discovered where Irene Adler hid the photograph by implementing a clever plan. He arranged for Watson to wait outside Briony Lodge with a smoke rocket. Holmes then engineered a quarrel that led to chaos, during which he pretended to be a clergyman and was "struck," allowing him to be carried inside the house. When the smoke rocket was deployed, Adler reacted by rushing to her hiding place to retrieve the photograph, revealing its location to Holmes.

However, he failed to recover it the next morning because by then, Irene Adler had already left the country with her new husband, taking the photograph with her. They departed on the five-fifteen train to the Continent, which was confirmed by an elderly maid-servant who admitted Holmes and the King the following morning (A Scandal in Bohemia— Sections 9 and 13).
      Cited: 05_scandal_in_bohemia.txt
      Expected: 05_scandal_in_bohemia.txt
      Facts matched: 2/5
      Latency: 3577ms

  Source-match: 3/3
```
