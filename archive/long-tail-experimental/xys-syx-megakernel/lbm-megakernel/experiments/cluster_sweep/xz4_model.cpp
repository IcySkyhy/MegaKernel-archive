// Independent Cartesian ownership enumeration; no CUDA or production decoder.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "../../include/layout_config.h"
static const int dx[19]={0,0,0,1,-1,0,0,1,-1,1,-1,0,0,0,0,1,1,-1,-1};
static const int dy[19]={0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0};
static const int dz[19]={0,0,0,0,0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
static bool physical(int x,int y,int z) {
    const int s=x+PADDED_X*(y+PADDED_Y*z);
    return s>=0&&s<PADDED_X*PADDED_Y*SIZE_Z&&s%PADDED_X<SIZE_X;
}
int main() {
    std::array<uint64_t,15> count{};
    uint64_t slots=0,cores=0;
    for(int bz=0;bz<38;bz+=2)for(int by=0;by<15;++by)for(int bx=0;bx<8;bx+=2) {
        const int ox=16*bx,oy=8*by,oz=4*bz;
        std::array<unsigned char,3400> producer{};
        std::vector<unsigned char> writers(19*2048,0);
        for(int z=-1;z<=8;++z)for(int y=-1;y<=8;++y)for(int x=-1;x<=32;++x) {
            const int rxowner=x<=15?0:1,rzowner=z<=3?0:1,rank=rxowner+2*rzowner;
            const bool valid=physical(ox+x,oy+y,oz+z);
            ++producer[x+1+34*(y+1+10*(z+1))];++count[rank];if(valid)++count[4+rank];
            for(int q=0;q<19;++q) {
                const int rx=x+dx[q],ry=y+dy[q],rz=z+dz[q];
                if(rx<0||rx>=32||ry<0||ry>=8||rz<0||rz>=8||
                   ox+rx>=SIZE_X||oy+ry>=SIZE_Y||oz+rz>=SIZE_Z)continue;
                const int tx=rx<16?0:1,tz=rz<4?0:1;
                const bool crossx=tx!=rxowner,crossz=tz!=rzowner,remote=crossx||crossz;
                ++writers[q*2048+rx+32*(ry+8*rz)];
                ++count[valid?(remote?9:8):(remote?11:10)];
                if(remote) {
                    ++count[crossx?(crossz?14:12):13];
                    if(crossx)assert((x==15&&dx[q]==1)||(x==16&&dx[q]==-1));
                    if(crossz)assert((z==3&&dz[q]==1)||(z==4&&dz[q]==-1));
                    if(crossx&&crossz)assert(q>=15&&q<=18); // ET/EB/WT/WB.
                }
            }
        }
        for(auto v:producer)assert(v==1);
        // Inverse r-cq lookup verifies both source ownership and the forward write.
        for(int z=0;z<8;++z)for(int y=0;y<8;++y)for(int x=0;x<32;++x) {
            bool live=ox+x<SIZE_X&&oy+y<SIZE_Y&&oz+z<SIZE_Z;cores+=live;
            for(int q=0;q<19;++q) {
                const int px=x-dx[q],py=y-dy[q],pz=z-dz[q];
                assert(producer[px+1+34*(py+1+10*(pz+1))]==1);
                const int rank=(px<=15?0:1)+2*(pz<=3?0:1);
                assert(rank>=0&&rank<4);
                assert(writers[q*2048+x+32*(y+8*z)]==(live?1:0));slots+=live;
            }
        }
    }
    for(int r=0;r<4;++r)assert(count[r]==969000);
    assert(cores==2160000&&slots==41040000);
    assert(count[14]>0&&count[12]+count[13]+count[14]==count[9]+count[11]);
    printf("{\"passed\":true,\"cores\":%llu,\"slots\":%llu,\"counts\":[",(unsigned long long)cores,(unsigned long long)slots);
    for(int i=0;i<15;++i)printf("%s%llu",i?",":"",(unsigned long long)count[i]);puts("]}");
}
