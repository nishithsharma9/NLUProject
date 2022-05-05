class ToxicityTransformer:
  def __init__(self,toxicClassifierModelSpacy,toxicSpanModelSpacy):
    self.toxicClassifierModelSpacy = deepcopy(toxicClassifierModelSpacy);
    self.toxicSpanModelSpacy = deepcopy(toxicSpanModelSpacy);
  
  def predict(self,input):
    doc1 = self.toxicSpanModelSpacy(input)
    doc2 = self.toxicClassifierModelSpacy(input)
    maskedOutput = input
    for ent in doc1.ents:
      maskedOutput=maskedOutput.replace(ent.string,"[MASK]")
    return {"classification":doc2.cats,"span":doc1.ents, "output":maskedOutput}
