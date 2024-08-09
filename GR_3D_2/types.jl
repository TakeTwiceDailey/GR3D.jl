# Alias for 4 vectors
const FourVector = Vec{4,Data.Number}

# Alias for 3 vector
const ThreeVector = Vec{3,Data.Number}

# Alias for symmetric 2 tensor
const TwoTensor = SymmetricSecondOrderTensor{2,Data.Number,3}

# Alias for symmetric 3 tensor
const ThreeTensor = SymmetricSecondOrderTensor{3,Data.Number,6}

# Alias for non-symmetric 4 tensor
const FourTensor = SecondOrderTensor{4,Data.Number,16}

# Alias for symmetric 4 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

# Alias for symmetric 4th rank tensor
const Projection = SymmetricFourthOrderTensor{4,Data.Number}

# Alias for non-symmetric 4th rank tensor
const NSProjection = FourthOrderTensor{4,Data.Number}

const SurfaceProjector = Mat{3,2,Data.Number}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor = Tensor{Tuple{4,@Symmetry{4,4}},Data.Number,3,40}

# Alias for 3 dimensional metric derivatives and Christoffel Symbols
const Symmetric3rdOrderTensor3 = Tensor{Tuple{3,@Symmetry{3,3}},Data.Number,3,18}

# Struct for main memory and Runge-Kutta algorithms that holds all state vector variables
struct SavedFourVector <: FieldVector{5,StateTensor}
    g::StateTensor  # metric tensor
    dx::StateTensor # x-derivative
    dy::StateTensor # y-derivative
    dz::StateTensor # z-derivative
    P::StateTensor  # normal projection derivative
end

# Struct for main memory and Runge-Kutta algorithms that holds all state vector variables
struct StateVector <: FieldVector{5,StateTensor}
    g::StateTensor  # metric tensor
    dx::StateTensor # x-derivative
    dy::StateTensor # y-derivative
    dz::StateTensor # z-derivative
    P::StateTensor  # normal projection derivative
end

math_operators = [:+,:-,:*]
# Define math operators for StateVector (Used in Runge-Kutta and Dissipation)
for op in math_operators
    @eval import Base.$op
    @eval @inline function $op(A::StateVector,B::StateVector)
        g  = @. $op(A.g,B.g)
        dx = @. $op(A.dx,B.dx)
        dy = @. $op(A.dy,B.dy)
        dz = @. $op(A.dz,B.dz)
        P  = @. $op(A.P,B.P)
        return StateVector(g,dx,dy,dz,P)
    end

    @eval @inline function $op(a::Number,B::StateVector)
        g  = $op(a,B.g)
        dx = $op(a,B.dx)
        dy = $op(a,B.dy)
        dz = $op(a,B.dz)
        P  = $op(a,B.P)
        return StateVector(g,dx,dy,dz,P)
    end

    @eval @inline $op(B::StateVector,a::Number) = $op(a,B)
end

@eval @inline function Base.:/(B::StateVector,a::Number)
    g  = B.g/a
    dx = B.dx/a
    dy = B.dy/a
    dz = B.dz/a
    P  = B.P/a
    return StateVector(g,dx,dy,dz,P)
end

@inline Base.:-(A::StateVector) = -1*A

@inline function Base.zero(::Type{StateVector})
    g  = zero(StateTensor)
    return StateVector(g,g,g,g,g)
end

# Some convenience values for these types
const NaNFV   = FourVector((NaN,NaN,NaN,NaN))
const Zero4   = zero(FourVector)
const ZeroST  = zero(StateTensor)
const MinST   = StateTensor((-1,0,0,0,1,0,0,1,0,1))
const MinSV   = StateVector(MinST,ZeroST,ZeroST,ZeroST,ZeroST)
const NaNST   = StateTensor((NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN))
const ZeroSV  = zero(StateVector)
const NaNSV   = StateVector(NaNST,NaNST,NaNST,NaNST,NaNST)