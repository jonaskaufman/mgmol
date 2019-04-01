// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// Written by J.-L. Fattebert, D. Osei-Kuffuor and I.S. Dunn.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Ion.h"
#include "MGmol_blas1.h"

#include <iomanip>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace std;

unsigned short isqrt(unsigned value)
{
    return static_cast<unsigned short>(sqrt(static_cast<double>(value)));
}

static unsigned int _nlproj_gid = 0;
static unsigned int _index      = 0;

Ion::Ion(const Species& species, const string& name, const double crds[3],
    const double velocity[3], const bool lock)
    : name_(name),
      species_(species),
      index_(_index),
      nlproj_gid_(_nlproj_gid),
      kbproj_(species)
{
    assert(name.size() > 0);

    _index++;
    _nlproj_gid += nProjectors();

    init(crds, velocity, lock);
};

Ion::Ion(const Species& species, const string& name, const double crds[3],
    const double velocity[3], const unsigned int index,
    const unsigned int nlproj_gid, const bool lock)
    : name_(name),
      species_(species),
      index_(index),
      nlproj_gid_(nlproj_gid),
      kbproj_(species)
{
    assert(name.size() > 0);

    kbproj_.initCenter(crds);

    init(crds, velocity, lock);
}

Ion::Ion(const Species& species, IonData data)
    : name_(data.ion_name),
      species_(species),
      index_(data.index),
      nlproj_gid_(data.nlproj_id),
      kbproj_(species)
{
    kbproj_.initCenter(data.current_position);

    const bool lock = data.atmove ? false : true;
    init(data.current_position, data.velocity, lock);
}

// initialize data members
void Ion::init(const double crds[3], const double velocity[3], const bool lock)
{
    for (short i = 0; i < 3; i++)
    {
        position_[i]     = crds[i];
        old_position_[i] = crds[i];
        lpot_start_[i]   = 0;
        lstart_[i]       = 0.;
        force_[i]        = 0.;
        velocity_[i]     = velocity[i];
    }
    locked_ = lock;
#if DEBUG
    (*MPIdata::sout) << " position:" << position_[0] << "," << position_[1]
                     << "," << position_[2] << endl;
    if (locked_) (*MPIdata::sout) << name_ << " locked" << endl;
#endif
    here_   = false;
    map_nl_ = false;
    map_l_  = false;

    // initialize random state seed
    int idx             = index_ + 1; // shift by 1 to avoid divide by zero
    unsigned short root = isqrt(idx);
    unsigned short rem  = (short)(idx % root);
    rand_state_[0]      = root;
    rand_state_[1]      = root;
    rand_state_[2]      = rem;
};

void Ion::setup()
{
    kbproj_.setup(position_);

    map_nl_ = kbproj_.overlapPE();
}

void Ion::bcast(MPI_Comm comm)
{
#ifdef USE_MPI
    MPI_Bcast(old_position_, 3, MPI_DOUBLE, 0, comm);
    MPI_Bcast(position_, 3, MPI_DOUBLE, 0, comm);
    MPI_Bcast(force_, 3, MPI_DOUBLE, 0, comm);
    MPI_Bcast(velocity_, 3, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&locked_, 1, MPI_CHAR, 0, comm);
#else
    return;
#endif
}

void Ion::set_lstart(const int i, const double cell_origin, const double hgrid)
{
    // t1 = nb of nodes between the boundary and crds
    double t1 = (position_[i] - cell_origin) / hgrid;

    // get the integral part of t1 in ic
    double t2;
    t1     = modf(t1, &t2);
    int ic = (int)t2;
    if (t1 > 0.5) ic++;
    if (t1 < -0.5) ic--;

    lpot_start_[i] = ic - (species_.dim_l() >> 1);
    lstart_[i]     = cell_origin + hgrid * lpot_start_[i];
    //(*MPIdata::sout)<<"lpot_start_[i]="<<lpot_start_[i]<<endl;
    //(*MPIdata::sout)<<"lstart_[i]    ="<<lstart_[i]<<endl;
    //(*MPIdata::sout)<<"species_.dim_l()="<<species_.dim_l()<<endl;
}

double Ion::minimage(const Ion& ion2, const double cell[3],
    const short periodic[3], double xc[3]) const
{
    for (short i = 0; i < 3; i++)
    {
        assert(cell[i] > 0.);
        xc[i] = position_[i] - ion2.position(i);

        if (periodic[i])
        {
            const double half_lattice = 0.5 * cell[i];
            xc[i]                     = fmod(xc[i], cell[i]);
            if (xc[i] > half_lattice)
                xc[i] -= cell[i];
            else if (xc[i] < -half_lattice)
                xc[i] += cell[i];
            assert(xc[i] <= half_lattice);
        }
    }

    const double rmin = sqrt(xc[0] * xc[0] + xc[1] * xc[1] + xc[2] * xc[2]);
    return rmin;
}

double Ion::distance(const Ion& ion2, double* xc, double* yc, double* zc) const
{
    *xc = position_[0] - ion2.position(0);
    *yc = position_[1] - ion2.position(1);
    *zc = position_[2] - ion2.position(2);

    const double r = (*xc) * (*xc) + (*yc) * (*yc) + (*zc) * (*zc);

    return sqrt(r);
}

double Ion::minimage(
    const Ion& ion2, const double cell[3], const short periodic[3]) const
{
    double xc[3] = { 0., 0., 0. };

    return minimage(ion2, cell, periodic, xc);
}

double Ion::distance(
    const Ion& ion2, const double ax, const double ay, const double az) const
{
    assert(ax > 0.);
    assert(ay > 0.);
    assert(az > 0.);

    const double dx = position_[0] - ion2.position(0);
    const double dy = position_[1] - ion2.position(1);
    const double dz = position_[2] - ion2.position(2);
    const double x  = fmod(dx, ax);
    const double y  = fmod(dy, ay);
    const double z  = fmod(dz, az);
    const double r  = x * x + y * y + z * z;

    return sqrt(r);
}

double Ion::minimage(
    const double point[3], const double cell[3], const short periodic[3]) const
{
    double xc[3];
    for (short i = 0; i < 3; i++)
    {
        assert(cell[i] > 0.);
        xc[i] = position_[i] - point[i];

        if (periodic[i] == 1)
        {
            const double half_lattice = 0.5 * cell[i];
            xc[i]                     = fmod(xc[i], cell[i]);
            if (xc[i] > half_lattice)
                xc[i] -= cell[i];
            else if (xc[i] < -half_lattice)
                xc[i] += cell[i];
            assert(xc[i] <= half_lattice);
        }
    }

    const double rmin = sqrt(xc[0] * xc[0] + xc[1] * xc[1] + xc[2] * xc[2]);
    return rmin;
}

bool operator<(Ion A, Ion B) { return (A.position(0) < B.position(0)); }

bool operator==(Ion A, Ion B) { return (&A == &B); }

void Ion::printPosition(ostream& os) const
{
    os << "$$ ";
    if (locked_)
        os << "*";
    else
        os << " ";
    os << setw(4) << name_ << setw(10) << setprecision(4) << fixed
       << position_[0] << setw(10) << position_[1] << setw(10) << position_[2]
       << endl;
}

void Ion::printPositionAndForce(ostream& os) const
{
    os << "## ";
    if (locked_)
        os << "*";
    else
        os << " ";
    os << setw(4) << name_ << setiosflags(ios::right) << setw(10)
       << setprecision(4) << fixed << position_[0] << setw(10) << position_[1]
       << setw(10) << position_[2] << setprecision(7) << scientific << setw(16)
       << force_[0] << setw(16) << force_[1] << setw(16) << force_[2] << endl;
}

void Ion::setIonData()
{
    idata_.ion_name   = name_;
    idata_.atomic_num = atomic_number();
    idata_.index      = index_;
    idata_.nlproj_id  = nlproj_gid_;
    idata_.pmass      = getMass();

    if (locked_)
        idata_.atmove = 0;
    else
        idata_.atmove = 1;

    for (int pos = 0; pos < 3; pos++)
    {
        idata_.rand_state[pos]       = rand_state_[pos];
        idata_.initial_position[pos] = 0.;
        idata_.old_position[pos]     = old_position_[pos];
        idata_.current_position[pos] = position_[pos];
        idata_.velocity[pos]         = velocity_[pos];
        idata_.force[pos]            = force_[pos];
    }
}

void Ion::setFromIonData(const IonData& data)
{
    // random state
    setRandomState(data.rand_state[0], data.rand_state[1], data.rand_state[2]);
    // previous position
    setOldPosition(
        data.old_position[0], data.old_position[1], data.old_position[2]);
    // velocity
    setVelocity(data.velocity[0], data.velocity[1], data.velocity[2]);
    // force
    setForce(data.force[0], data.force[1], data.force[2]);
}

// get gids (row index for <KB,Psi> matrix) for all the projectors
// of Ion
void Ion::getGidsNLprojs(vector<int>& gids) const
{
    // get global indexes by adding nlproj_gid_
    gids.clear();

    short nproj = nProjectors();
    for (short i = 0; i < nproj; ++i)
    {
        const int gid = nlproj_gid_ + i;
        gids.push_back(gid);
    }
}

// get sign factor for <KB,Psi> matrix for all the projectors
// of Ion
void Ion::getKBsigns(vector<short>& kbsigns) const
{
    kbproj_.getKBsigns(kbsigns);
}

void Ion::getKBcoeffs(vector<double>& coeffs) { kbproj_.getKBcoeffs(coeffs); }

void Ion::get_Ai(vector<int>& Ai, const int gdim, const short dir) const
{
    int idx = lpot_start(dir);

    int diml = species_.dim_l();
    diml     = min(diml, gdim - 1);
    Ai.resize(diml);

    const int sizeAi = (int)Ai.size();
    for (int ix = 0; ix < sizeAi; ix++)
    {
        Ai[ix] = idx % gdim;

        while (Ai[ix] < 0)
        {
            Ai[ix] += gdim; // periodic BC
        }
        assert(Ai[ix] >= 0);
        assert(Ai[ix] < gdim);

        idx++;
    }
}

double Ion::energyDiff(
    Ion& ion, const double lattice[3], const short bc[3]) const
{
    const double r = minimage(ion, lattice, bc);
    assert(r > 0.);

    const double t1 = sqrt(getRC() * getRC() + ion.getRC() * ion.getRC());

    return getZion() * ion.getZion() * erfc(r / t1) / r;
}

//This is where the local atomic potential is set, as well as the compensating
//charge.
void Ion::addContributionToVnucAndRhoc(POTDTYPE* vnuc,
                                       RHODTYPE* rhoc,
                                       const pb::Grid& mygrid,
                                       const short bc[3])
{
    assert(map_l_);

    const double h0 = mygrid.hgrid(0);
    const double h1 = mygrid.hgrid(1);
    const double h2 = mygrid.hgrid(2);

    Vector3D ll(mygrid.ll(0), mygrid.ll(1), mygrid.ll(2));

    const int dim0  = mygrid.dim(0);
    const int dim1  = mygrid.dim(1);
    const int dim2  = mygrid.dim(2);

    const int inc0 = dim1 * dim2;

    // min./max. indices of subdomain mesh
    const int ilow = mygrid.istart(0);
    const int jlow = mygrid.istart(1);
    const int klow = mygrid.istart(2);
    const int ihi  = ilow + dim0 - 1;
    const int jhi  = jlow + dim1 - 1;
    const int khi  = klow + dim2 - 1;

    const Vector3D position(position_[0], position_[1], position_[2]);

    // Generate indices where potential is not zero
    vector<vector<int>> Ai;
    Ai.resize(3);
    get_Ai(Ai[0], mygrid.gdim(0), 0);
    get_Ai(Ai[1], mygrid.gdim(1), 1);
    get_Ai(Ai[2], mygrid.gdim(2), 2);
    const int dimlx = Ai[0].size();
    const int dimly = Ai[1].size();
    const int dimlz = Ai[2].size();

    const double rc = species_.rc();
    const double lr = species_.lradius();
    assert(rc > 1.e-8);

    const double pi3half = M_PI * sqrt(M_PI);
    const double rcnorm = rc * rc * rc * pi3half;
    assert(rcnorm > 1.e-8);
    const double alpha   = getZion() / rcnorm;
    const double inv_rc2 = 1. / (rc * rc);

    const RadialInter& lpot(getLocalPot());
#if DEBUG
    int icount        = 0;
    double charge_ion = 0.;
#endif

    double zc         = lstart_[2];

    //triple loop over indices where potential is not zero
    for (int iz = 0; iz < dimlz; iz++)
    {
        double yc       = lstart_[1];
        const int ai2iz = Ai[2][iz];
        const int ivecz = ai2iz % dim2;

        if ((ai2iz >= klow) && (ai2iz <= khi))
        for (int iy = 0; iy < dimly; iy++)
        {
            double xc       = lstart_[0];
            const int ai1iy = Ai[1][iy];
            const int ivecy = dim2 * (ai1iy % dim1);

            if ((ai1iy >= jlow) && (ai1iy <= jhi))
            for (int ix = 0; ix < dimlx; ix++)
            {
                const int ai0ix = Ai[0][ix];
                if ((ai0ix >= ilow) && (ai0ix <= ihi))
                {
                    const int ivec = inc0 * (ai0ix % dim0)
                                     + ivecy
                                     + ivecz;
                    assert(ivec < mygrid.size());

                    Vector3D vc(xc, yc, zc);
                    const double r
                        = vc.minimage(position, ll, bc);
                    const double r2 = r * r;

                    const double tmp = alpha * exp(-r2 * inv_rc2);
                    rhoc[ivec] += tmp;
#if DEBUG
                    charge_ion += tmp;
#endif
                    if (r < lr)
                    {
#if 0
                        vnuc[ivec] += 
                            get_trilinval(xc,yc,zc,h0,h1,h2,position,ll,lpot);
#else
                        vnuc[ivec] += lpot.cubint(r);
#endif
                    }
#if DEBUG
                    icount++;
#endif
                }

                xc += h0;

            } // end loop over ix

            yc += h1;

        } // end loop over iy

        zc += h2;

    } // end loop over iz

#if DEBUG
    cout << setprecision(10);
    cout << " icount=" << icount << endl;
    cout << " charge compensating Ion=" << mygrid.vel() * charge_ion
        << endl;
#endif

}

