from enum import IntEnum

from .Variable import BasicVariable, UFuncVariable, SumVariable, DifferenceVariable
from .VariableBase import VariableBase
from simonplot.typing.Protocols import VariableProtocol

from simonpy.coordinates import xyz_to_eta_phi
import numpy as np

class SplittingClass(IntEnum):
    other = 0
    gTOggTOgg = 1
    gTOggTOqq = 2
    gTOqqTOqg = 3
    qTOqgTOgg = 4
    qTOqgTOqq = 5
    qTOgqTOgq = 6    

class SplittingPdgIdsToSplittingClass(VariableBase):

    def __init__(self,
                 pdgid1 : VariableProtocol | str,
                 pdgid2 : VariableProtocol | str,
                 pdgid3 : VariableProtocol | str,
                 pdgid4 : VariableProtocol | str,
                 pdgid5 : VariableProtocol | str,
                 pdgid6 : VariableProtocol | str):
        if isinstance(pdgid1, str):
            pdgid1 = BasicVariable(pdgid1)
        if isinstance(pdgid2, str):
            pdgid2 = BasicVariable(pdgid2)
        if isinstance(pdgid3, str):
            pdgid3 = BasicVariable(pdgid3)
        if isinstance(pdgid4, str):
            pdgid4 = BasicVariable(pdgid4)
        if isinstance(pdgid5, str):
            pdgid5 = BasicVariable(pdgid5)
        if isinstance(pdgid6, str):
            pdgid6 = BasicVariable(pdgid6)

        self._pdgid1 = pdgid1
        self._pdgid2 = pdgid2
        self._pdgid3 = pdgid3
        self._pdgid4 = pdgid4   
        self._pdgid5 = pdgid5
        self._pdgid6 = pdgid6

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self):
        return False
    
    @property
    def columns(self):
        return list(set(
            self._pdgid1.columns +
            self._pdgid2.columns +
            self._pdgid3.columns +
            self._pdgid4.columns +
            self._pdgid5.columns +
            self._pdgid6.columns
        ))
    
    def evaluate(self, dataset, cut):
        pdgid1 = self._pdgid1.evaluate(dataset, cut)
        pdgid2 = self._pdgid2.evaluate(dataset, cut)
        pdgid3 = self._pdgid3.evaluate(dataset, cut)
        pdgid4 = self._pdgid4.evaluate(dataset, cut)
        pdgid5 = self._pdgid5.evaluate(dataset, cut)
        pdgid6 = self._pdgid6.evaluate(dataset, cut)

        isG1 = (pdgid1 == 21)
        isG2 = (pdgid2 == 21)
        isG3 = (pdgid3 == 21)
        isG4 = (pdgid4 == 21)
        isG5 = (pdgid5 == 21)
        isG6 = (pdgid6 == 21)

        isQ1 = (np.abs(pdgid1) <= 6) & (np.abs(pdgid1) > 0)
        isQ2 = (np.abs(pdgid2) <= 6) & (np.abs(pdgid2) > 0)
        isQ3 = (np.abs(pdgid3) <= 6) & (np.abs(pdgid3) > 0)
        isQ4 = (np.abs(pdgid4) <= 6) & (np.abs(pdgid4) > 0)
        isQ5 = (np.abs(pdgid5) <= 6) & (np.abs(pdgid5) > 0)
        isQ6 = (np.abs(pdgid6) <= 6) & (np.abs(pdgid6) > 0)

        # first splitting
        gTOgg1 = isG1 & isG2 & isG3
        gTOqq1 = isG1 & isQ2 & isQ3
        qTOqg1 = isQ1 & ( (isQ2 & isG3) | (isG2 & isQ3) )

        gTOgg2 = isG4 & isG5 & isG6
        gTOqq2 = isG4 & isQ5 & isQ6
        qTOqg2 = isQ4 & ( (isQ5 & isG6) | (isG5 & isQ6) )

        gTOggTOgg = gTOgg1 & gTOgg2
        gTOggTOqq = gTOgg1 & gTOqq2
        gTOqqTOqg = gTOqq1 & qTOqg2
        qTOqgTOgg = qTOqg1 & gTOgg2
        qTOqgTOqq = qTOqg1 & gTOqq2
        qTOgqTOgq = qTOqg1 & qTOqg2

        splitting_class = np.where(
            gTOggTOgg, SplittingClass.gTOggTOgg,
            np.where(
                gTOggTOqq, SplittingClass.gTOggTOqq,
                np.where(
                    gTOqqTOqg, SplittingClass.gTOqqTOqg,
                    np.where(
                        qTOqgTOgg, SplittingClass.qTOqgTOgg,
                        np.where(
                            qTOqgTOqq, SplittingClass.qTOqgTOqq,
                            np.where(
                                qTOgqTOgq, SplittingClass.qTOgqTOgq,
                                SplittingClass.other
                            )
                        )
                    )
                )
            )
        )

    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        pdgid1 = self._pdgid1.to_pyarrow_expression()
        pdgid2 = self._pdgid2.to_pyarrow_expression()
        pdgid3 = self._pdgid3.to_pyarrow_expression()
        pdgid4 = self._pdgid4.to_pyarrow_expression()
        pdgid5 = self._pdgid5.to_pyarrow_expression()
        pdgid6 = self._pdgid6.to_pyarrow_expression()

        if pdgid1 is None or pdgid2 is None or pdgid3 is None or pdgid4 is None or pdgid5 is None or pdgid6 is None:
            raise ValueError("Cannot convert SplittingPdgIdsToSplittingClass to pyarrow expression because one or more of the pdgid variables is not convertible to pyarrow expression.")
        
        pdgid1 = pc.abs(pdgid1)
        pdgid2 = pc.abs(pdgid2)
        pdgid3 = pc.abs(pdgid3)
        pdgid4 = pc.abs(pdgid4)
        pdgid5 = pc.abs(pdgid5)
        pdgid6 = pc.abs(pdgid6)

        isG1 = pc.equal(pdgid1, 21)
        isG2 = pc.equal(pdgid2, 21)
        isG3 = pc.equal(pdgid3, 21)
        isG4 = pc.equal(pdgid4, 21)
        isG5 = pc.equal(pdgid5, 21)
        isG6 = pc.equal(pdgid6, 21)

        isQ1 = pc.and_kleene(pc.less_equal(pdgid1, 6), pc.greater(pdgid1, 0))
        isQ2 = pc.and_kleene(pc.less_equal(pdgid2, 6), pc.greater(pdgid2, 0))
        isQ3 = pc.and_kleene(pc.less_equal(pdgid3, 6), pc.greater(pdgid3, 0))
        isQ4 = pc.and_kleene(pc.less_equal(pdgid4, 6), pc.greater(pdgid4, 0))
        isQ5 = pc.and_kleene(pc.less_equal(pdgid5, 6), pc.greater(pdgid5, 0))
        isQ6 = pc.and_kleene(pc.less_equal(pdgid6, 6), pc.greater(pdgid6, 0))

        gTOgg1 = pc.and_kleene(
            isG1,
            pc.and_kleene(
                isG2,
                isG3
            )
        )
        gTOqq1 = pc.and_kleene(
            isG1,
            pc.and_kleene(
                isQ2,
                isQ3
            )
        )
        qTOqg1 = pc.and_kleene(
            isQ1,
            pc.or_kleene(
                pc.and_kleene(
                    isQ2,
                    isG3
                ),
                pc.and_kleene(
                    isG2,
                    isQ3
                )
            )
        )

        gTOgg2 = pc.and_kleene(
            isG4,
            pc.and_kleene(
                isG5,
                isG6
            )
        )   
        gTOqq2 = pc.and_kleene(
            isG4,
            pc.and_kleene(
                isQ5,
                isQ6
            )
        )
        qTOqg2 = pc.and_kleene(
            isQ4,
            pc.or_kleene(
                pc.and_kleene(
                    isQ5,
                    isG6
                ),
                pc.and_kleene(
                    isG5,
                    isQ6
                )
            )
        )

        gTOggTOgg = pc.and_kleene(gTOgg1, gTOgg2)
        gTOggTOqq = pc.and_kleene(gTOgg1, gTOqq2)
        gTOqqTOqg = pc.and_kleene(gTOqq1, qTOqg2)
        qTOqgTOgg = pc.and_kleene(qTOqg1, gTOgg2)
        qTOqgTOqq = pc.and_kleene(qTOqg1, gTOqq2)
        qTOgqTOgq = pc.and_kleene(qTOqg1, qTOqg2)

        return pc.if_else(
            gTOggTOgg, pc.scalar(SplittingClass.gTOggTOgg),
            pc.if_else(
                gTOggTOqq, pc.scalar(SplittingClass.gTOggTOqq),
                pc.if_else(
                    gTOqqTOqg, pc.scalar(SplittingClass.gTOqqTOqg),
                    pc.if_else(
                        qTOqgTOgg, pc.scalar(SplittingClass.qTOqgTOgg),
                        pc.if_else(
                            qTOqgTOqq, pc.scalar(SplittingClass.qTOqgTOqq),
                            pc.if_else(
                                qTOgqTOgq, pc.scalar(SplittingClass.qTOgqTOgq),
                                pc.scalar(SplittingClass.other)
                            )
                        )
                    )
                )
            )
        )

    @property
    def key(self):
        return "SplittingPdgIdsToSplittingClass(%s_%s_%s_%s_%s_%s)" % (
            self._pdgid1.key,
            self._pdgid2.key,
            self._pdgid3.key,
            self._pdgid4.key,
            self._pdgid5.key,
            self._pdgid6.key
        )
    
    def __eq__(self, other):
        if type(other) is not SplittingPdgIdsToSplittingClass:
            return False
        
        return (self._pdgid1 == other._pdgid1 and
                self._pdgid2 == other._pdgid2 and
                self._pdgid3 == other._pdgid3 and
                self._pdgid4 == other._pdgid4 and
                self._pdgid5 == other._pdgid5 and
                self._pdgid6 == other._pdgid6)
    
    def set_collection_name(self, collection_name):
        self._pdgid1.set_collection_name(collection_name)
        self._pdgid2.set_collection_name(collection_name)
        self._pdgid3.set_collection_name(collection_name)
        self._pdgid4.set_collection_name(collection_name)
        self._pdgid5.set_collection_name(collection_name)
        self._pdgid6.set_collection_name(collection_name)

class RelativeResolutionVariable(VariableBase):
    def __init__(self, gen : VariableProtocol | str, reco : VariableProtocol | str):
        if isinstance(gen, str):
            gen = BasicVariable(gen)
        if isinstance(reco, str):
            reco = BasicVariable(reco)

        self._gen = gen
        self._reco = reco

    @property
    def _natural_centerline(self):
        return 0.0
    
    @property
    def prebinned(self) -> bool:
        return False

    @property
    def columns(self):
        return list(set(self._gen.columns + self._reco.columns))
    
    def evaluate(self, dataset, cut):
        gen = self._gen.evaluate(dataset, cut)
        reco = self._reco.evaluate(dataset, cut)
        return (reco - gen) / gen

    @property
    def key(self):
        return "%s_minus_%s_over_%s"%(self._reco.key, self._gen.key, self._gen.key)
    
    def __eq__(self, other):
        if type(other) is not RelativeResolutionVariable:
            return False
        return self._gen == other._gen and self._reco == other._reco

    def set_collection_name(self, collection_name):
        self._gen.set_collection_name(collection_name)
        self._reco.set_collection_name(collection_name)

    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        reco = self._reco.to_pyarrow_expression()
        gen = self._gen.to_pyarrow_expression()
        assert(reco is not None)
        assert(gen is not None)
        return pc.divide(
            pc.subtract(
                reco,
                gen
            ),
            gen
        )

class Magnitude3dVariable(VariableBase):
    def __init__(self, xvar: VariableProtocol | str, yvar: VariableProtocol | str, zvar: VariableProtocol | str):
        import numpy as np

        if isinstance(xvar, str):
            xvar = BasicVariable(xvar)
        if isinstance(yvar, str):
            yvar = BasicVariable(yvar)
        if isinstance(zvar, str):
            zvar = BasicVariable(zvar)

        self._xvar = xvar
        self._yvar = yvar
        self._zvar = zvar

        self._x2var = UFuncVariable(self._xvar, np.square)
        self._y2var = UFuncVariable(self._yvar, np.square)
        self._z2var = UFuncVariable(self._zvar, np.square)

        self.r2var = SumVariable(
            SumVariable(self._x2var, self._y2var),
            self._z2var
        )

        self._rvar = UFuncVariable(self.r2var, np.sqrt)
    
    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False

    @property
    def columns(self):
        return list(set(
            self._xvar.columns +
            self._yvar.columns +
            self._zvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self._rvar.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "sqrt(%s^2 + %s^2 + %s^2)"%(self._xvar.key, self._yvar.key, self._zvar.key)

    def __eq__(self, other):
        if type(other) is not Magnitude3dVariable:
            return False
        return (self._xvar == other._xvar and
                self._yvar == other._yvar and
                self._zvar == other._zvar)

    def set_collection_name(self, collection_name):
        self._rvar.set_collection_name(collection_name)
        self._xvar.set_collection_name(collection_name)
        self._yvar.set_collection_name(collection_name)
        self._zvar.set_collection_name(collection_name)

    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        x = self._xvar.to_pyarrow_expression()
        y = self._yvar.to_pyarrow_expression()
        z = self._zvar.to_pyarrow_expression()
        assert(x is not None)
        assert(y is not None)
        assert(z is not None)
        return pc.sqrt(
            pc.add(
                pc.add(
                    pc.power(x, 2),
                    pc.power(y, 2)
                ),
                pc.power(z, 2)
            )
        )

class Magnitude2dVariable(VariableBase):
    def __init__(self, xvar: VariableProtocol | str, yvar: VariableProtocol | str):
        import numpy as np

        if isinstance(xvar, str):
            xvar = BasicVariable(xvar)
        if isinstance(yvar, str):
            yvar = BasicVariable(yvar)

        self._xvar = xvar
        self._yvar = yvar

        self._x2var = UFuncVariable(self._xvar, np.square)
        self._y2var = UFuncVariable(self._yvar, np.square)

        self._r2var = SumVariable(self._x2var, self._y2var)
        self._rvar = UFuncVariable(self._r2var, np.sqrt)
    
    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(
            self._xvar.columns +
            self._yvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self._rvar.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "sqrt(%s^2 + %s^2)"%(self._xvar.key, self._yvar.key)

    def __eq__(self, other):
        if type(other) is not Magnitude3dVariable:
            return False
        return (self._xvar == other._xvar and
                self._yvar == other._yvar)

    def set_collection_name(self, collection_name):
        self._rvar.set_collection_name(collection_name)
        self._xvar.set_collection_name(collection_name)
        self._yvar.set_collection_name(collection_name)

    def to_pyarrow_expression(self):  
        import pyarrow.compute as pc
        x = self._xvar.to_pyarrow_expression()
        y = self._yvar.to_pyarrow_expression()
        assert(x is not None)
        assert(y is not None)
        return pc.sqrt(
            pc.add(
                pc.power(x, 2),
                pc.power(y, 2)
            )
        )

class Distance3dVariable(VariableBase):
    def __init__(self, x1var: VariableProtocol | str, y1var: VariableProtocol | str, z1var: VariableProtocol | str, x2var: VariableProtocol | str, y2var: VariableProtocol | str, z2var: VariableProtocol | str):
        import numpy as np

        if isinstance(x1var, str):
            x1var = BasicVariable(x1var)
        if isinstance(y1var, str):
            y1var = BasicVariable(y1var)
        if isinstance(z1var, str):
            z1var = BasicVariable(z1var)
        if isinstance(x2var, str):
            x2var = BasicVariable(x2var)
        if isinstance(y2var, str):
            y2var = BasicVariable(y2var)
        if isinstance(z2var, str):
            z2var = BasicVariable(z2var)

        self._dxvar = DifferenceVariable(x1var, x2var)
        self._dyvar = DifferenceVariable(y1var, y2var)
        self._dzvar = DifferenceVariable(z1var, z2var)

        self.magnitude_var = Magnitude3dVariable(
            self._dxvar,
            self._dyvar,
            self._dzvar
        )

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(
            self._dxvar.columns +
            self._dyvar.columns +
            self._dzvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self.magnitude_var.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "Distance3D(%s_%s_%s - %s_%s_%s)"%(
            self._dxvar._var2.key,
            self._dyvar._var2.key,
            self._dzvar._var2.key,
            self._dxvar._var1.key,
            self._dyvar._var1.key,
            self._dzvar._var1.key
        )
    
    def __eq__(self, other):
        if type(other) is not Distance3dVariable:
            return False
        return (self._dxvar == other._dxvar and
                self._dyvar == other._dyvar and
                self._dzvar == other._dzvar)
    
    def set_collection_name(self, collection_name):
        self._dxvar.set_collection_name(collection_name)
        self._dyvar.set_collection_name(collection_name)
        self._dzvar.set_collection_name(collection_name)
        self.magnitude_var.set_collection_name(collection_name)
        
    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        return pc.sqrt(
            pc.add(
                pc.add(
                    pc.power(self._dxvar.to_pyarrow_expression(), 2),
                    pc.power(self._dyvar.to_pyarrow_expression(), 2)
                ),
                pc.power(self._dzvar.to_pyarrow_expression(), 2)
            )
        )
    
class DeltaPhiVariable(VariableBase):
    def __init__(self, phi1 : VariableProtocol | str, phi2: VariableProtocol | str):
        if isinstance(phi1, str):
            phi1 = BasicVariable(phi1)
        if isinstance(phi2, str):
            phi2 = BasicVariable(phi2)

        self._phi1 = phi1
        self._phi2 = phi2

    @property
    def _natural_centerline(self):
        return 0.0

    @property
    def prebinned(self) -> bool:
        return False

    @property
    def columns(self):
        return list(set(self._phi1.columns + self._phi2.columns))

    @property
    def key(self):
        return "DeltaPhi(%s_%s)" % (self._phi1.key, self._phi2.key)

    def __eq__(self, other):
        if type(other) is not DeltaPhiVariable:
            return False

        return self._phi1 == other._phi1 and self._phi2 == other._phi2

    def set_collection_name(self, collection_name):
        self._phi1.set_collection_name(collection_name)
        self._phi2.set_collection_name(collection_name)

    def evaluate(self, dataset, cut):
        import numpy as np

        phi1val = self._phi1.evaluate(dataset, cut)
        phi2val = self._phi2.evaluate(dataset, cut)

        dphi = phi1val - phi2val
        dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        return dphi

    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        import numpy as np

        phi1 = self._phi1.to_pyarrow_expression()
        phi2 = self._phi2.to_pyarrow_expression()
        assert(phi1 is not None)
        assert(phi2 is not None)

        dphi = pc.subtract(phi1, phi2)
        dphi = pc.if_else(pc.greater(dphi, np.pi), pc.subtract(dphi, 2*np.pi), dphi)
        dphi = pc.if_else(pc.less(dphi, -np.pi), pc.add(dphi, 2*np.pi), dphi)
        return dphi

class DeltaRVariable(VariableBase):
    def __init__(self, eta1 : VariableProtocol | str, phi1: VariableProtocol | str, eta2 : VariableProtocol | str, phi2: VariableProtocol | str):
        if isinstance(eta1, str):
            eta1 = BasicVariable(eta1)
        if isinstance(phi1, str):
            phi1 = BasicVariable(phi1)
        if isinstance(eta2, str):
            eta2 = BasicVariable(eta2)
        if isinstance(phi2, str):
            phi2 = BasicVariable(phi2)

        self._eta1 = eta1
        self._phi1 = phi1
        self._eta2 = eta2
        self._phi2 = phi2

        self._deta = DifferenceVariable(self._eta1, self._eta2)
        self._dphi = DeltaPhiVariable(self._phi1, self._phi2)
        self._dr = Magnitude2dVariable(self._deta, self._dphi)

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False

    @property
    def columns(self):
        return list(set(
            self._eta1.columns +
            self._phi1.columns +
            self._eta2.columns +
            self._phi2.columns
        ))

    @property
    def key(self):
        return "DeltaR(%s_%s__%s_%s)" % (self._eta1.key, self._phi1.key, self._eta2.key, self._phi2.key)

    def __eq__(self, other):
        if type(other) is not DeltaRVariable:
            return False

        return (self._eta1 == other._eta1 and
                self._phi1 == other._phi1 and
                self._eta2 == other._eta2 and
                self._phi2 == other._phi2)

    def set_collection_name(self, collection_name):
        self._eta1.set_collection_name(collection_name)
        self._phi1.set_collection_name(collection_name)
        self._eta2.set_collection_name(collection_name)
        self._phi2.set_collection_name(collection_name)
        self._dr.set_collection_name(collection_name)

    def evaluate(self, dataset, cut):
        return self._dr.evaluate(dataset, cut)
    
    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        import numpy as np

        eta1 = self._eta1.to_pyarrow_expression()
        phi1 = self._phi1.to_pyarrow_expression()
        eta2 = self._eta2.to_pyarrow_expression()
        phi2 = self._phi2.to_pyarrow_expression()
        assert(eta1 is not None)
        assert(phi1 is not None)
        assert(eta2 is not None)
        assert(phi2 is not None)    

        deta = pc.subtract(eta1, eta2)
        dphi = pc.subtract(phi1, phi2)
        dphi = pc.if_else(pc.greater(dphi, np.pi), pc.subtract(dphi, 2*np.pi), dphi)
        dphi = pc.if_else(pc.less(dphi, -np.pi), pc.add(dphi, 2*np.pi), dphi)

        return pc.sqrt(
            pc.add(
                pc.power(deta, 2),
                pc.power(dphi, 2)
            )
        )

class Distance2dVariable(VariableBase):
    def __init__(self, x1var, y1var, x2var, y2var):
        import numpy as np

        self._dxvar = DifferenceVariable(x1var, x2var)
        self._dyvar = DifferenceVariable(y1var, y2var)

        self.magnitude_var = Magnitude2dVariable(
            self._dxvar,
            self._dyvar
        )
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property
    def columns(self):
        return list(set(
            self._dxvar.columns +
            self._dyvar.columns
        ))  
    
    def evaluate(self, dataset, cut):
        return self.magnitude_var.evaluate(dataset, cut)
    
    @property
    def key(self):
        return "Distance2D(%s_%s - %s_%s)"%(
            self._dxvar._var2.key,
            self._dyvar._var2.key,
            self._dxvar._var1.key,
            self._dyvar._var1.key
        )
    
    def __eq__(self, other):
        if type(other) is not Distance2dVariable:
            return False
        return (self._dxvar == other._dxvar and
                self._dyvar == other._dyvar)
    
    def set_collection_name(self, collection_name):
        self._dxvar.set_collection_name(collection_name)
        self._dyvar.set_collection_name(collection_name)
        self.magnitude_var.set_collection_name(collection_name)

    def to_pyarrow_expression(self):
        import pyarrow.compute as pc
        return pc.sqrt(
            pc.add(
                pc.power(self._dxvar.to_pyarrow_expression(), 2),
                pc.power(self._dyvar.to_pyarrow_expression(), 2)
            )
        )

class EtaFromXYZVariable(VariableBase):
    def __init__(self, x : VariableProtocol | str, y: VariableProtocol | str, z: VariableProtocol | str):
        if isinstance(x, str):
            x = BasicVariable(x)
        if isinstance(y, str):
            y = BasicVariable(y)
        if isinstance(z, str):
            z = BasicVariable(z)

        self._x = x
        self._y = y
        self._z = z

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property 
    def columns(self):
        return list(set(self._x.columns + self._y.columns + self._z.columns))
    
    @property
    def key(self):
        return "ETA(%s_%s_%s)" % (self._x.key, self._y.key, self._z.key)
    
    def __eq__(self, other):
        if type(other) is not EtaFromXYZVariable:
            return False
        
        return (self._x == other._x and 
                self._y == other._y and
                self._z == other._z)
    
    def set_collection_name(self, collection_name):
        self._x.set_collection_name(collection_name)
        self._y.set_collection_name(collection_name)
        self._z.set_collection_name(collection_name)

    def evaluate(self, dataset, cut):
        xval = self._x.evaluate(dataset, cut)
        yval = self._y.evaluate(dataset, cut)
        zval = self._z.evaluate(dataset, cut)

        return xyz_to_eta_phi(xval, yval, zval)[0]
    
    def to_pyarrow_expression(self):
        raise NotImplementedError("EtaFromXYZVariable does not currently support to_pyarrow_expression. This is because the transformation from XYZ to eta/phi is non-trivial to express in pyarrow, and we have not yet implemented it. If you need this functionality, please open an issue or submit a pull request implementing it.")

class PhiFromXYZVariable(VariableBase):
    def __init__(self, x : VariableProtocol | str, y: VariableProtocol | str, z: VariableProtocol | str):
        if isinstance(x, str):
            x = BasicVariable(x)
        if isinstance(y, str):
            y = BasicVariable(y)
        if isinstance(z, str):
            z = BasicVariable(z)

        self._x = x
        self._y = y
        self._z = z

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def prebinned(self) -> bool:
        return False
    
    @property 
    def columns(self):
        return list(set(self._x.columns + self._y.columns + self._z.columns))
    
    @property
    def key(self):
        return "PHI(%s_%s_%s)" % (self._x.key, self._y.key, self._z.key)
    
    def __eq__(self, other):
        if type(other) is not EtaFromXYZVariable:
            return False
        
        return (self._x == other._x and 
                self._y == other._y and
                self._z == other._z)
    
    def set_collection_name(self, collection_name):
        self._x.set_collection_name(collection_name)
        self._y.set_collection_name(collection_name)
        self._z.set_collection_name(collection_name)

    def evaluate(self, dataset, cut):
        xval = self._x.evaluate(dataset, cut)
        yval = self._y.evaluate(dataset, cut)
        zval = self._z.evaluate(dataset, cut)

        return xyz_to_eta_phi(xval, yval, zval)[1]

    def to_pyarrow_expression(self):
        raise NotImplementedError("PhiFromXYZVariable does not currently support to_pyarrow_expression. This is because the transformation from XYZ to eta/phi is non-trivial to express in pyarrow, and we have not yet implemented it. If you need this functionality, please open an issue or submit a pull request implementing it.")